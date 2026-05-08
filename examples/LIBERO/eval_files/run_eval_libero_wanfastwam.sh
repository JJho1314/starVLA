#!/bin/bash
# Evaluate Wan_FastWAM with FastWAM-official release ckpt on LIBERO (4 suites).
#
# Same architecture as eval_all_parallel.sh but adapted for our wanfastwam server:
#   - Server side: starVLA env, builds Wan_FastWAM and loads FastWAM ckpt via key mapping.
#   - Client side: libero env (mujoco + sim), connects to server via websocket.
#
# Defaults match jhe724's HPC3 layout. Override via env vars to use elsewhere.
#
# Usage:
#   bash examples/LIBERO/eval_files/run_eval_libero_wanfastwam.sh
#   # or with single suite:
#   SUITES="libero_spatial" bash examples/LIBERO/eval_files/run_eval_libero_wanfastwam.sh

set -u

STARVLA_DIR=${STARVLA_DIR:-/data/user/jhe724/workspace/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/data/user/jhe724/workspace/LIBERO}
STARVLA_PYTHON=${STARVLA_PYTHON:-/data/user/jhe724/.conda/envs/starVLA/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/data/user/jhe724/.conda/envs/libero/bin/python}

# === FastWAM-aligned config + ckpt ===
CONFIG_YAML=${CONFIG_YAML:-${STARVLA_DIR}/examples/LIBERO/train_files/starvla_wanfastwam_libero.yaml}
FASTWAM_CKPT=${FASTWAM_CKPT:-/data/user/jhe724/workspace/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt}
WAN22_PATH=${WAN22_PATH:-/data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers}
DATASET_STATS=${DATASET_STATS:-/data/user/jhe724/workspace/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_wanfastwam_release}
RUN_DIR=${STARVLA_DIR}/playground/eval_logs/${RUN_TAG}
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd "${STARVLA_DIR}"
export PYTHONPATH=${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}
export LIBERO_HOME LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

HOST=127.0.0.1
PORTS=${PORTS:-"6694 6695"}
GPUS=${GPUS:-"0 1"}
N_TRIALS=${N_TRIALS:-50}
N_WORKERS_PER_SUITE=${N_WORKERS_PER_SUITE:-5}
TASKS_PER_WORKER=${TASKS_PER_WORKER:-2}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

PORT_ARR=(${PORTS})
GPU_ARR=(${GPUS})
SUITE_ARR=(${SUITES})

start_server() {
  local gpu="$1" port="$2"
  echo "[$(date +%H:%M:%S)] launching wanfastwam server on GPU ${gpu} port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu} ${STARVLA_PYTHON} deployment/model_server/server_wanfastwam.py \
      --config_yaml "${CONFIG_YAML}" \
      --fastwam_ckpt "${FASTWAM_CKPT}" \
      --wan22_path "${WAN22_PATH}" \
      --port ${port} \
      --use_bf16 \
      --idle_timeout -1 \
      > "${LOG_DIR}/server_gpu${gpu}_p${port}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port="$1"
  for i in $(seq 1 600); do
    if (echo > /dev/tcp/${HOST}/${port}) 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] server :${port} READY"; return 0
    fi
    sleep 1
  done
  echo "[$(date +%H:%M:%S)] server :${port} did NOT come up in 600s"; return 1
}

# 1) Boot servers
declare -a SERVER_PIDS=()
for i in "${!PORT_ARR[@]}"; do
  pid=$(start_server "${GPU_ARR[$i]}" "${PORT_ARR[$i]}")
  SERVER_PIDS+=("$pid")
done

cleanup() {
  echo "[$(date +%H:%M:%S)] cleanup: killing servers ${SERVER_PIDS[*]}"
  for p in "${SERVER_PIDS[@]}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for p in "${PORT_ARR[@]}"; do
  wait_for_server "$p" || { echo "server boot failed"; exit 1; }
done

# 2) Launch eval clients per suite
declare -a CLIENT_PIDS=()
PORT_RR_IDX=0
for suite in "${SUITE_ARR[@]}"; do
  for w in $(seq 0 $((N_WORKERS_PER_SUITE - 1))); do
    start=$((w * TASKS_PER_WORKER))
    end=$((start + TASKS_PER_WORKER))
    port=${PORT_ARR[$((PORT_RR_IDX % ${#PORT_ARR[@]}))]}
    PORT_RR_IDX=$((PORT_RR_IDX + 1))
    log="${LOG_DIR}/client_${suite}_w${w}.log"
    echo "[$(date +%H:%M:%S)] eval ${suite} worker=${w} task_id=[${start},${end}) port=${port}"
    ${LIBERO_PYTHON} examples/LIBERO/eval_files/eval_libero.py \
        --task_suite_name ${suite} \
        --task_id_start ${start} \
        --task_id_end ${end} \
        --worker_id ${w} \
        --num_trials_per_task ${N_TRIALS} \
        --pretrained_path "${DATASET_STATS}" \
        --host ${HOST} \
        --port ${port} \
        --video_out_path "${RUN_DIR}/${suite}" \
        --job_name "wanfastwam_${suite}_w${w}" \
        > "${log}" 2>&1 &
    CLIENT_PIDS+=($!)
  done
done

# 3) Wait for all clients
echo "[$(date +%H:%M:%S)] waiting for ${#CLIENT_PIDS[@]} clients ..."
for p in "${CLIENT_PIDS[@]}"; do
  wait "$p"
done
echo "[$(date +%H:%M:%S)] all clients done."

# 4) Aggregate per-worker summaries -> per-suite SR
echo "[$(date +%H:%M:%S)] aggregating SR ..."
for suite in "${SUITE_ARR[@]}"; do
  ${STARVLA_PYTHON} - <<PYEOF
import json, glob, os
files = sorted(glob.glob("${RUN_DIR}/${suite}/_summary_w*.json"))
total_ep, total_succ = 0, 0
per_task = {}
for f in files:
    s = json.load(open(f))
    total_ep += s["total_episodes"]
    total_succ += s["total_successes"]
    per_task.update({int(k): v for k, v in s["per_task"].items()})
sr = total_succ / total_ep if total_ep else 0.0
print(f"  ${suite}: SR = {sr*100:.2f}%  ({total_succ}/{total_ep} over {len(per_task)} tasks)")
out = {"suite": "${suite}", "SR": sr, "total_episodes": total_ep,
       "total_successes": total_succ, "per_task": per_task}
json.dump(out, open(os.path.join("${RUN_DIR}", f"${suite}_aggregate.json"), "w"), indent=2)
PYEOF
done
echo "[$(date +%H:%M:%S)] DONE.  Logs + summaries in ${RUN_DIR}"
