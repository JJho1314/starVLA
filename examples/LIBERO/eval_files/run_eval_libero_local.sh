#!/bin/bash
# Local LIBERO eval (single GPU, single suite, configurable workers).
# Use this to quickly validate the pipeline end-to-end before launching full 4-suite eval.
#
# Defaults match LFT-W02 layout. Override via env vars.
#
# Usage:
#   bash examples/LIBERO/eval_files/run_eval_libero_local.sh
#   SUITE=libero_spatial N_WORKERS=2 N_TRIALS=5 bash examples/LIBERO/eval_files/run_eval_libero_local.sh

set -u

STARVLA_DIR=${STARVLA_DIR:-/data/LFT-W02_data/junjie/VLA_WM/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/data/LFT-W02_data/junjie/LIBERO}
STARVLA_PYTHON=${STARVLA_PYTHON:-/data/LFT-W02_data/.conda/envs/starVLA/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/data/LFT-W02_data/.conda/envs/libero/bin/python}

# === Eval run dir built via make_eval_run_dir.py ===
RUN_DIR_PRETRAINED=${RUN_DIR_PRETRAINED:-/data/LFT-W02_data/junjie/eval_runs/wanfastwam_release}
PRETRAINED_PT="${RUN_DIR_PRETRAINED}/checkpoints/libero_uncond_2cam224.pt"
CONFIG_YAML="${RUN_DIR_PRETRAINED}/config.yaml"
FASTWAM_CKPT=${FASTWAM_CKPT:-/data/LFT-W02_data/junjie/weights/fastwam/libero_uncond_2cam224.pt}
WAN22_PATH=${WAN22_PATH:-/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers}

# === Eval scope ===
SUITE=${SUITE:-libero_object}            # one of libero_spatial/object/goal/10
N_WORKERS=${N_WORKERS:-2}
TASKS_PER_WORKER=${TASKS_PER_WORKER:-5}   # libero suites have 10 tasks each
N_TRIALS=${N_TRIALS:-5}                   # episodes per task; 5 is fast smoke; 50 is final
GPU=${GPU:-0}
PORT=${PORT:-7090}

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_local_${SUITE}}
RUN_OUT=${STARVLA_DIR}/playground/eval_logs/${RUN_TAG}
LOG_DIR="${RUN_OUT}/logs"
mkdir -p "${LOG_DIR}"

cd "${STARVLA_DIR}"
export PYTHONPATH=${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}
export LIBERO_HOME LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

HOST=127.0.0.1

# 1) Boot wanfastwam server
echo "[$(date +%H:%M:%S)] launching wanfastwam server on GPU ${GPU} port ${PORT}"
CUDA_VISIBLE_DEVICES=${GPU} ${STARVLA_PYTHON} deployment/model_server/server_wanfastwam.py \
    --config_yaml "${CONFIG_YAML}" \
    --fastwam_ckpt "${FASTWAM_CKPT}" \
    --wan22_path "${WAN22_PATH}" \
    --port ${PORT} \
    --use_bf16 \
    --idle_timeout -1 \
    --dataset_stats "${RUN_DIR_PRETRAINED}/dataset_statistics.json" \
    > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "  server PID=$SERVER_PID"

cleanup() {
  echo "[$(date +%H:%M:%S)] cleanup: killing server $SERVER_PID"
  kill -TERM $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 2) Wait for server
echo "[$(date +%H:%M:%S)] waiting for server :${PORT} (build + ckpt load takes ~30s) ..."
for i in $(seq 1 600); do
  if (echo > /dev/tcp/${HOST}/${PORT}) 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] server :${PORT} READY (after ${i}s)"; break
  fi
  sleep 1
  if [[ $i -eq 600 ]]; then
    echo "[$(date +%H:%M:%S)] server boot TIMEOUT, see ${LOG_DIR}/server.log"
    tail -30 "${LOG_DIR}/server.log"
    exit 1
  fi
done

# 3) Launch N_WORKERS clients on the suite
declare -a CLIENT_PIDS=()
for w in $(seq 0 $((N_WORKERS - 1))); do
  start=$((w * TASKS_PER_WORKER))
  end=$((start + TASKS_PER_WORKER))
  log="${LOG_DIR}/client_w${w}.log"
  echo "[$(date +%H:%M:%S)] eval ${SUITE} worker=${w} task_id=[${start},${end}) trials=${N_TRIALS}"
  ${LIBERO_PYTHON} examples/LIBERO/eval_files/eval_libero.py \
      --args.task-suite-name ${SUITE} \
      --args.task-id-start ${start} \
      --args.task-id-end ${end} \
      --args.worker-id ${w} \
      --args.num-trials-per-task ${N_TRIALS} \
      --args.pretrained-path "${PRETRAINED_PT}" \
      --args.host ${HOST} \
      --args.port ${PORT} \
      --args.video-out-path "${RUN_OUT}/${SUITE}" \
      --args.job-name "wanfastwam_${SUITE}_w${w}" \
      > "${log}" 2>&1 &
  CLIENT_PIDS+=($!)
done

echo "[$(date +%H:%M:%S)] waiting for ${#CLIENT_PIDS[@]} clients ..."
for p in "${CLIENT_PIDS[@]}"; do
  wait "$p"
done

# 4) Aggregate
${STARVLA_PYTHON} - <<PYEOF
import json, glob, os
files = sorted(glob.glob("${RUN_OUT}/${SUITE}/_summary_w*.json"))
total_ep, total_succ = 0, 0
per_task = {}
for f in files:
    s = json.load(open(f))
    total_ep += s["total_episodes"]
    total_succ += s["total_successes"]
    per_task.update({int(k): v for k, v in s["per_task"].items()})
sr = (total_succ / total_ep) if total_ep else 0.0
print(f"\n${SUITE}: SR = {sr*100:.2f}%  ({total_succ}/{total_ep} over {len(per_task)} tasks)")
for tid in sorted(per_task.keys()):
    pt = per_task[tid]
    task_sr = pt["successes"] / pt["episodes"] if pt["episodes"] else 0.0
    print(f"  task {tid:3d}: {task_sr*100:5.1f}%  ({pt['successes']}/{pt['episodes']})  {pt.get('description','')[:60]}")
out = {"suite": "${SUITE}", "SR": sr, "total_episodes": total_ep,
       "total_successes": total_succ, "per_task": per_task}
json.dump(out, open(os.path.join("${RUN_OUT}", f"${SUITE}_aggregate.json"), "w"), indent=2)
PYEOF

echo "[$(date +%H:%M:%S)] DONE.  Logs in ${RUN_OUT}"
