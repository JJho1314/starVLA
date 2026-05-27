#!/bin/bash
# Low-concurrency LIBERO eval for WanMetaQueryFastWAM checkpoints.
# It keeps the policy server count small and runs suites sequentially to avoid
# MuJoCo/EGL framebuffer failures from too many concurrent offscreen clients.
set -u

STARVLA_DIR=${STARVLA_DIR:-/data/LFT-W02_data/junjie/VLA_WM/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/data/LFT-W02_data/junjie/LIBERO}
STARVLA_PYTHON=${STARVLA_PYTHON:-/data/LFT-W02_data/.conda/envs/starVLA/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/data/LFT-W02_data/.conda/envs/libero/bin/python}

FWALIGN_RUN_DIR=${FWALIGN_RUN_DIR:-${STARVLA_DIR}/playground/Checkpoints/20260521_231811_309267_libero_WanMetaQueryFastWAM_30k_bs128}
CKPT=${CKPT:-${FWALIGN_RUN_DIR}/checkpoints/steps_20000_pytorch_model.pt}
RUN_DIR=$(dirname "$(dirname "$CKPT")")
LOG_DIR=${LOG_DIR:-${RUN_DIR}/eval_logs_lowconcurrency}
RESULTS_SUBDIR=${RESULTS_SUBDIR:-results_lowconcurrency}
mkdir -p "$LOG_DIR"

cd "$STARVLA_DIR" || exit 1
export PYTHONPATH=${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}
export LIBERO_HOME LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export TOKENIZERS_PARALLELISM=false

export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-${STARVLA_FASTWAM_REPO_PATH}/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

HOST=${HOST:-127.0.0.1}
PORTS_CSV=${PORTS_CSV:-6694}
GPUS_CSV=${GPUS_CSV:-1}
IFS=',' read -r -a PORTS <<< "$PORTS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ ${#PORTS[@]} -ne ${#GPUS[@]} ]]; then
  echo "[fatal] PORTS_CSV and GPUS_CSV must have the same number of entries" >&2
  exit 1
fi

N_TRIALS=${N_TRIALS:-50}
SUITES_STR=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
read -r -a SUITES <<< "$SUITES_STR"
N_WORKERS_PER_SUITE=${N_WORKERS_PER_SUITE:-1}
TASKS_PER_WORKER=${TASKS_PER_WORKER:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
BATCH_WAIT_MS=${BATCH_WAIT_MS:-20}

for p in "$CKPT" "$RUN_DIR/config.yaml" "$RUN_DIR/dataset_statistics.json" \
         "${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" \
         "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio"; do
  [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done

echo "[ok] RUN_DIR=${RUN_DIR}"
echo "[ok] CKPT=${CKPT}"
echo "[ok] SUITES=${SUITES_STR} N_TRIALS=${N_TRIALS}"
echo "[ok] GPUS=${GPUS_CSV} PORTS=${PORTS_CSV} workers/suite=${N_WORKERS_PER_SUITE} tasks/worker=${TASKS_PER_WORKER}"

start_server() {
  local gpu="$1" port="$2"
  echo "[$(date +%H:%M:%S)] launching MetaQuery server on GPU ${gpu} port ${port} (batch_size=${BATCH_SIZE})"
  CUDA_VISIBLE_DEVICES=${gpu} "${STARVLA_PYTHON}" deployment/model_server/server_wanfastwam_starvla.py \
    --run_dir "${RUN_DIR}" \
    --ckpt_path "${CKPT}" \
    --port "${port}" \
    --use_bf16 \
    --idle_timeout -1 \
    --batch_size "${BATCH_SIZE}" \
    --batch_wait_ms "${BATCH_WAIT_MS}" \
    > "${LOG_DIR}/server_gpu${gpu}_p${port}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port="$1"
  for _ in $(seq 1 3600); do
    if (echo > /dev/tcp/${HOST}/${port}) 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] server :${port} READY"
      return 0
    fi
    sleep 1
  done
  echo "[$(date +%H:%M:%S)] server :${port} did NOT come up" >&2
  return 1
}

declare -a SERVER_PIDS=()
for i in "${!PORTS[@]}"; do
  pid=$(start_server "${GPUS[$i]}" "${PORTS[$i]}")
  SERVER_PIDS+=("$pid")
done
trap 'echo "[$(date +%H:%M:%S)] cleanup; killing servers ${SERVER_PIDS[*]}"; kill -9 ${SERVER_PIDS[*]} 2>/dev/null' EXIT

for p in "${PORTS[@]}"; do
  wait_for_server "$p" || exit 1
done

folder=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
client_idx=0

for suite in "${SUITES[@]}"; do
  out_root="${RUN_DIR}/${RESULTS_SUBDIR}/${suite}/${folder}"
  mkdir -p "${out_root}"
  echo "[$(date +%H:%M:%S)] === suite ${suite} start ==="

  declare -a WORKER_PIDS=()
  for w in $(seq 0 $((N_WORKERS_PER_SUITE - 1))); do
    s=$((w * TASKS_PER_WORKER))
    e=$(((w + 1) * TASKS_PER_WORKER))
    # Pin worker's MuJoCo EGL rendering to the same GPU as its policy server,
    # otherwise MuJoCo defaults to GPU 0 → 7+ workers all rendering on GPU 0
    # while GPU 1 only carries the policy server.
    port_idx=$((client_idx % ${#PORTS[@]}))
    port=${PORTS[$port_idx]}
    egl_gpu=${GPUS[$port_idx]}
    client_idx=$((client_idx + 1))
    seed=$((42 + w * 1000))
    log="${LOG_DIR}/eval_${suite}_w${w}.log"
    echo "[$(date +%H:%M:%S)] start ${suite} worker ${w} tasks[${s},${e}) :${port} EGL=${egl_gpu}"
    (
      MUJOCO_EGL_DEVICE_ID=${egl_gpu} CUDA_VISIBLE_DEVICES=${egl_gpu} \
      "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "${CKPT}" \
        --args.host "${HOST}" \
        --args.port "${port}" \
        --args.task-suite-name "${suite}" \
        --args.num-trials-per-task "${N_TRIALS}" \
        --args.task-id-start "${s}" \
        --args.task-id-end "${e}" \
        --args.worker-id "${w}" \
        --args.seed "${seed}" \
        --args.video-out-path "${out_root}/w${w}" \
        > "${log}" 2>&1
      echo "[$(date +%H:%M:%S)] DONE ${suite} worker ${w} rc=$?"
    ) &
    WORKER_PIDS+=("$!")
  done

  suite_rc=0
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || suite_rc=$?
  done
  if [[ "$suite_rc" -ne 0 ]]; then
    echo "[fatal] suite ${suite} failed with rc=${suite_rc}" >&2
    exit "$suite_rc"
  fi

  "${STARVLA_PYTHON}" - "$out_root" "$suite" <<'PY'
import glob, json, sys
out_root, suite = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{out_root}/w*/_summary_w*.json"))
ep = 0
sc = 0
per_task = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    ep += d["total_episodes"]
    sc += d["total_successes"]
    per_task.update(d.get("per_task", {}))
rate = sc / ep if ep else 0.0
print(f"{suite}: episodes={ep} successes={sc} rate={rate:.4f} ({100 * rate:.1f}%) workers={len(files)}")
with open(f"{out_root}/_aggregate.json", "w") as fh:
    json.dump({"suite": suite, "total_episodes": ep, "total_successes": sc, "rate": rate, "per_task": per_task}, fh, indent=2)
PY
  echo "[$(date +%H:%M:%S)] === suite ${suite} done ==="
done

echo "=== ALL DONE ==="
