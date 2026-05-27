#!/bin/bash
# LIBERO eval — joint-FAITHFUL inference path.
# Sibling of `eval_all_parallel_fwalign.sh`. Only differences:
#   - launches `server_wanfastwam_starvla_joint_faithful.py` instead of
#     `server_wanfastwam_starvla.py` (rebinds predict_action → predict_action_joint).
#   - per-server cost is higher (NSW × full MoT forward, not 1 prefill + NSW action-only),
#     so the default per-server BATCH_SIZE is conservative.
#
# Intended for ckpts trained with `framework.fastwam.mot_attention_mode: joint`.
# The standard (mask-only / KV-cache-reuse) joint launcher is
# `eval_all_parallel_fwalign.sh` via `run_eval_fwalign_joint_local.sh`; this
# launcher is the strictly-faithful counterpart that mirrors FastWAMJoint.infer_action.
set -u

STARVLA_DIR=/data/LFT-W02_data/junjie/VLA_WM/starVLA
LIBERO_HOME=/data/LFT-W02_data/junjie/LIBERO
STARVLA_PYTHON=/data/LFT-W02_data/.conda/envs/starVLA/bin/python
LIBERO_PYTHON=/data/LFT-W02_data/.conda/envs/libero/bin/python

CKPT=${CKPT:-${STARVLA_DIR}/playground/Checkpoints/1229_libero4in1_wm4a_cosmopredict2gr00t/checkpoints/steps_80000_pytorch_model.pt}
RUN_DIR=$(dirname $(dirname "$CKPT"))   # .../1229_libero4in1_wm4a_cosmopredict2gr00t
LOG_DIR="$RUN_DIR/eval_logs_joint_faithful"
mkdir -p "$LOG_DIR"

cd "$STARVLA_DIR"
export PYTHONPATH=${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}
export LIBERO_HOME LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

HOST=127.0.0.1
PORTS=(6695)
GPUS=(1)
N_TRIALS=${N_TRIALS:-50}
# 10 tasks per suite, 5 workers × 2 tasks each
N_WORKERS_PER_SUITE=${N_WORKERS_PER_SUITE:-5}
TASKS_PER_WORKER=${TASKS_PER_WORKER:-2}
SUITES_STR=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
read -r -a SUITES <<<"$SUITES_STR"

# Joint-faithful inference is ~NSW× more expensive than the standard path
# (1 full MoT forward per denoise step, no KV cache reuse) → keep per-server
# concurrency lower. 5 client workers × 2 servers = 10; default batch_size=5.
BATCH_SIZE=${BATCH_SIZE:-5}
BATCH_WAIT_MS=${BATCH_WAIT_MS:-20}

start_server() {
  local gpu="$1" port="$2"
  echo "[$(date +%H:%M:%S)] launching joint-faithful server on GPU ${gpu} port ${port} (batch_size=${BATCH_SIZE} wait=${BATCH_WAIT_MS}ms)"
  CUDA_VISIBLE_DEVICES=${gpu} ${STARVLA_PYTHON} deployment/model_server/server_wanfastwam_starvla_joint_faithful.py \
      --run_dir ${RUN_DIR} \
      --ckpt_path ${CKPT} \
      --port ${port} \
      --use_bf16 \
      --idle_timeout -1 \
      --batch_size ${BATCH_SIZE} \
      --batch_wait_ms ${BATCH_WAIT_MS} \
      > "${LOG_DIR}/server_gpu${gpu}_p${port}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port="$1"
  for i in $(seq 1 3600); do
    if (echo > /dev/tcp/${HOST}/${port}) 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] server :${port} READY"; return 0
    fi
    sleep 1
  done
  echo "[$(date +%H:%M:%S)] server :${port} did NOT come up in 3600s"; return 1
}

declare -a SERVER_PIDS=()
for i in 0; do
  pid=$(start_server "${GPUS[$i]}" "${PORTS[$i]}")
  SERVER_PIDS+=("$pid")
done
trap 'echo "[$(date +%H:%M:%S)] cleanup; killing joint-faithful servers ${SERVER_PIDS[*]}"; kill -9 ${SERVER_PIDS[*]} 2>/dev/null' EXIT

for p in "${PORTS[@]}"; do
  if ! wait_for_server "$p"; then
    echo "FATAL: server :${p} failed"
    exit 1
  fi
done

folder=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
folder="${folder}_jointfaithful"   # disambiguate from the cache-reuse path results

declare -a WORKER_PIDS=()
client_idx=0
for suite in "${SUITES[@]}"; do
  out_root="${RUN_DIR}/results/${suite}/${folder}"
  mkdir -p "${out_root}"
  for w in $(seq 0 $((N_WORKERS_PER_SUITE-1))); do
    s=$((w * TASKS_PER_WORKER))
    e=$(((w+1) * TASKS_PER_WORKER))
    port=${PORTS[0]}
    client_idx=$((client_idx+1))
    seed=$((42 + w * 1000 + RANDOM % 100))
    log="${LOG_DIR}/eval_${suite}_w${w}.log"
    echo "[$(date +%H:%M:%S)] start ${suite} worker ${w} tasks[${s},${e}) :${port} seed ${seed}"
    (
      ${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
          --args.pretrained-path ${CKPT} \
          --args.host ${HOST} \
          --args.port ${port} \
          --args.task-suite-name ${suite} \
          --args.num-trials-per-task ${N_TRIALS} \
          --args.task-id-start ${s} \
          --args.task-id-end ${e} \
          --args.worker-id ${w} \
          --args.seed ${seed} \
          --args.video-out-path "${out_root}/w${w}" \
          > "${log}" 2>&1
      echo "[$(date +%H:%M:%S)] DONE ${suite} worker ${w} rc=$?"
    ) &
    WORKER_PIDS+=("$!")
  done
done

echo "Total joint-faithful workers launched: ${#WORKER_PIDS[@]}"

for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid"
done

echo "[$(date +%H:%M:%S)] all joint-faithful eval workers finished"

echo "=== AGGREGATE (joint-faithful) ==="
for suite in "${SUITES[@]}"; do
  out_root="${RUN_DIR}/results/${suite}/${folder}"
  ${STARVLA_PYTHON} - "$out_root" "$suite" <<'PY'
import json, sys, glob
out_root, suite = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{out_root}/w*/_summary_w*.json"))
ep = 0; sc = 0; per_task = {}
for f in files:
    with open(f) as fh: d = json.load(fh)
    ep += d["total_episodes"]; sc += d["total_successes"]
    per_task.update(d.get("per_task", {}))
rate = sc / ep if ep else 0.0
print(f"{suite}: episodes={ep} successes={sc} rate={rate:.4f} ({100*rate:.1f}%) workers={len(files)}")
with open(f"{out_root}/_aggregate.json", "w") as fh:
    json.dump({"suite": suite, "total_episodes": ep, "total_successes": sc, "rate": rate, "per_task": per_task}, fh, indent=2)
PY
done
echo "=== ALL DONE (joint-faithful) ==="
