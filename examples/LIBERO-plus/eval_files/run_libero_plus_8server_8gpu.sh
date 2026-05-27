#!/bin/bash
# LIBERO-plus zeroshot eval with 8 model servers (1 per RTX 4090) + parallel clients.
# Each server holds 1 copy of WanFastWAM and batches inference across its clients
# (request batching = HIGH GPU utilization, unlike the previous in-process worker
# version where the GPU sat idle during mujoco env steps).
#
# Usage:
#   FWALIGN:  CKPT=/data3/junjie/fwalign_olabots_ckpts/final_model/pytorch_model.pt \
#             bash run_libero_plus_8server_8gpu.sh
#   FWJOINT:  CKPT=/data3/junjie/fwjoint_olabots_ckpts/final_model/pytorch_model.pt \
#             JOINT_FAITHFUL=1 bash run_libero_plus_8server_8gpu.sh
#
# Layout:
#   8 model servers, ports 6694..6701, GPUs 0..7
#   4 suites, ceil(N_TASKS / TASKS_PER_WORKER) workers per suite
#   Workers connect round-robin to ports → ~equal load per server
#   batch_size = 4 (each server batches up to 4 client requests per tick)
set -euo pipefail

##### Paths #####
STARVLA_DIR=${STARVLA_DIR:-/data3/junjie/starVLA}
LIBERO_PLUS_HOME=${LIBERO_PLUS_HOME:-/data3/junjie/LIBERO-plus}
CONDA_PY=${CONDA_PY:-/data3/junjie/envs/starvla_eval/bin/python}
CKPT=${CKPT:?CKPT must be set (e.g. /data3/junjie/fwalign_olabots_ckpts/final_model/pytorch_model.pt)}
RUN_DIR=$(dirname $(dirname "$CKPT"))
TEXT_CACHE=${TEXT_CACHE:-/data3/junjie/libero_plus_text_cache_lg1}
JOINT_FAITHFUL=${JOINT_FAITHFUL:-0}   # 1 → server dispatches predict_action_joint

if [[ "$JOINT_FAITHFUL" == "1" ]]; then
  TAG=fwjoint
else
  TAG=fwalign
fi
OUTPUT_DIR=${OUTPUT_DIR:-/data3/junjie/eval_runs/libero_plus_${TAG}_$(date +%Y%m%d_%H%M%S)}

##### FastWAM repo + DiffSynth + Wan ckpts (V8 parity) #####
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data3/junjie/FastWAM_official_clean}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-/data3/junjie/FastWAM_official_clean/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/data3/junjie}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# wand (motion_blur disturbance) needs libMagickWand from the conda env. Without
# this on LD_LIBRARY_PATH, wand raises ImportError and motion-blur tasks fall back
# to identity. ${CONDA_PREFIX:-...} resolves to the active starvla_eval env.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data3/junjie/envs/starvla_eval}/lib:${LD_LIBRARY_PATH:-}"
export LIBERO_HOME="$LIBERO_PLUS_HOME"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_PLUS_HOME}:${PYTHONPATH:-}"

##### Eval tuning #####
NUM_TRIALS=${NUM_TRIALS:-1}
# Each suite has ~2400-2600 tasks; with 8 workers/suite × 4 suites = 32 clients
# spread across 8 servers (~4 clients/server) we get sane GPU batching without
# OOMing the CPU on Python process startup (315 clients = 150GB+ RAM = swap death).
# NUM_GPUS: how many GPUs to use (from GPU 0 up). Fewer GPUs + more clients/server
# packs more concurrent requests onto each server → bigger batches → higher GPU
# util + more memory used + more CPU (mujoco) busy.
NUM_GPUS=${NUM_GPUS:-8}
WORKERS_PER_SUITE=${WORKERS_PER_SUITE:-8}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
# BATCH_SIZE should roughly match clients-per-server so each forward batches all
# in-flight requests. clients_per_server = (4 suites * WORKERS_PER_SUITE) / NUM_GPUS.
BATCH_SIZE=${BATCH_SIZE:-12}                 # server-side max batch
BATCH_WAIT_MS=${BATCH_WAIT_MS:-30}           # server-side max wait per batch

##### Boot #####
mkdir -p "$OUTPUT_DIR/server_logs" "$OUTPUT_DIR/client_logs"
echo "==== LIBERO-plus 8-server eval ===="
echo "  TAG          = $TAG"
echo "  CKPT         = $CKPT"
echo "  TEXT_CACHE   = $TEXT_CACHE"
echo "  OUTPUT_DIR   = $OUTPUT_DIR"
echo "  NUM_TRIALS   = $NUM_TRIALS  WORKERS_PER_SUITE = $WORKERS_PER_SUITE"
echo "  BATCH_SIZE   = $BATCH_SIZE  BATCH_WAIT_MS   = $BATCH_WAIT_MS"
echo "  JOINT_FAITHFUL = $JOINT_FAITHFUL"
echo

cd "$STARVLA_DIR"

HOST=127.0.0.1
ALL_PORTS=(${EVAL_PORTS:-6694 6695 6696 6697 6698 6699 6700 6701})
ALL_GPUS=(${EVAL_GPUS:-0 1 2 3 4 5 6 7})
# Slice to the first NUM_GPUS entries.
PORTS=("${ALL_PORTS[@]:0:$NUM_GPUS}")
GPUS=("${ALL_GPUS[@]:0:$NUM_GPUS}")
SUITES=(${EVAL_SUITES:-libero_spatial libero_object libero_goal libero_10})
echo "  NUM_GPUS     = $NUM_GPUS  (GPUs: ${GPUS[*]}  ports: ${PORTS[*]})"
echo "  clients/server ≈ $(( 4 * WORKERS_PER_SUITE / NUM_GPUS ))"
echo

start_server() {
  local gpu=$1 port=$2
  echo "[$(date +%H:%M:%S)] starting server gpu=$gpu port=$port"
  local extra=""
  [[ "$JOINT_FAITHFUL" == "1" ]] && extra="--joint_faithful"
  CUDA_VISIBLE_DEVICES=$gpu "$CONDA_PY" \
      deployment/model_server/server_wanfastwam_lg1_lp.py \
      --run_dir "$RUN_DIR" \
      --ckpt_path "$CKPT" \
      --text_cache_dir "$TEXT_CACHE" \
      --port $port --use_bf16 --idle_timeout -1 \
      --batch_size $BATCH_SIZE --batch_wait_ms $BATCH_WAIT_MS \
      $extra \
      >"$OUTPUT_DIR/server_logs/server_gpu${gpu}_p${port}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port=$1
  for _ in $(seq 1 1800); do
    if (echo > /dev/tcp/$HOST/$port) 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] server :${port} READY"
      return 0
    fi
    sleep 2
  done
  echo "FATAL: server :${port} timed out"
  return 1
}

declare -a SERVER_PIDS=()
for i in "${!GPUS[@]}"; do
  pid=$(start_server "${GPUS[$i]}" "${PORTS[$i]}")
  SERVER_PIDS+=("$pid")
done
trap 'echo "[$(date +%H:%M:%S)] cleanup; killing ${SERVER_PIDS[*]}"; kill -9 "${SERVER_PIDS[@]}" 2>/dev/null' EXIT

for p in "${PORTS[@]}"; do
  if ! wait_for_server "$p"; then
    echo "FATAL: server :$p never came up; aborting"
    exit 1
  fi
done

echo
echo "[$(date +%H:%M:%S)] all 8 servers READY; launching client workers ..."

suite_size() {
  CUDA_VISIBLE_DEVICES="" "$CONDA_PY" - <<PY 2>/dev/null | tail -1 | tr -dc '0-9'
from libero.libero import benchmark
print(benchmark.get_benchmark_dict()['$1']().n_tasks)
PY
}

declare -a WORKER_PIDS=()
client_idx=0
for suite in "${SUITES[@]}"; do
  n=$(suite_size "$suite")
  if [[ -z "$n" || "$n" -eq 0 ]]; then
    echo "WARN: failed to read n_tasks for $suite (got '$n'); skipping"
    continue
  fi
  num_workers=$WORKERS_PER_SUITE
  per_w=$(( (n + num_workers - 1) / num_workers ))
  echo "[$(date +%H:%M:%S)] $suite  n_tasks=$n  workers=$num_workers tasks_per_worker=$per_w"
  out_root="${OUTPUT_DIR}/${suite}"
  mkdir -p "$out_root"
  for w in $(seq 0 $((num_workers - 1))); do
    s=$((w * per_w))
    e=$(((w + 1) * per_w))
    [[ $e -gt $n ]] && e=$n
    [[ $s -ge $n ]] && continue
    # Round-robin to ports
    port=${PORTS[$((client_idx % NUM_GPUS))]}
    client_idx=$((client_idx + 1))
    seed=$((42 + w))
    log="${OUTPUT_DIR}/client_logs/${suite}_w${w}.log"
    # Rotate the GPU each client uses for mujoco EGL render across the 8 GPUs.
    # Without this all 32 clients default to GPU 0 for EGL and quickly fill its
    # 24 GB (each client takes 200-600 MiB for the EGL context). The server's
    # compute GPU is set at server launch — this only affects rendering.
    render_gpu=${GPUS[$((client_idx % NUM_GPUS))]}
    (
      CUDA_VISIBLE_DEVICES=$render_gpu \
      "$CONDA_PY" examples/LIBERO-plus/eval_files/eval_libero_plus_client.py \
          --args.pretrained-path "$CKPT" \
          --args.host $HOST --args.port $port \
          --args.task-suite-name "$suite" \
          --args.num-trials-per-task $NUM_TRIALS \
          --args.num-steps-wait $NUM_STEPS_WAIT \
          --args.task-id-start $s --args.task-id-end $e \
          --args.worker-id $w \
          --args.seed $seed \
          --args.video-out-path "$out_root/w${w}" \
          --args.no-save-video \
          >"$log" 2>&1
      rc=$?
      echo "[$(date +%H:%M:%S)] DONE $suite w$w rc=$rc"
    ) &
    WORKER_PIDS+=("$!")
  done
done

echo "[$(date +%H:%M:%S)] launched ${#WORKER_PIDS[@]} workers across ${#SUITES[@]} suites"

for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid" || true
done

echo "[$(date +%H:%M:%S)] all workers finished. Aggregating ..."
"$CONDA_PY" - <<PY
import glob, json, os
root = "${OUTPUT_DIR}"
by_suite = {}
for j in sorted(glob.glob(os.path.join(root, "*", "w*", "_summary_w*.json"))):
    d = json.load(open(j))
    s = d["task_suite_name"]
    bucket = by_suite.setdefault(s, {"total": 0, "ok": 0, "disturb": {}})
    bucket["total"] += d["total_episodes"]
    bucket["ok"] += d["total_successes"]
    for cat, v in (d.get("disturb_breakdown") or {}).items():
        b = bucket["disturb"].setdefault(cat, {"total_count": 0, "success_count": 0})
        b["total_count"] += v["total_count"]
        b["success_count"] += v["success_count"]
print("=== LIBERO-plus zeroshot (${TAG}) ===")
overall_n = overall_k = 0
for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    b = by_suite.get(s)
    if not b:
        continue
    n, k = b["total"], b["ok"]
    overall_n += n; overall_k += k
    print(f"{s:18s} {k}/{n}  {100*k/max(n,1):.2f}%")
    for cat in sorted(b["disturb"]):
        v = b["disturb"][cat]
        tot = v["total_count"]; ok = v["success_count"]
        print(f"    {cat:24s} {ok}/{tot}  {100*ok/max(tot,1):.2f}%")
print(f"{'OVERALL':18s} {overall_k}/{overall_n}  {100*overall_k/max(overall_n,1):.2f}%")
with open(os.path.join(root, "aggregate.json"), "w") as f:
    json.dump(by_suite, f, indent=2)
PY
echo "[$(date +%H:%M:%S)] aggregate.json written under $OUTPUT_DIR/"
