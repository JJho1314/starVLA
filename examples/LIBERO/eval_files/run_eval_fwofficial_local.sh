#!/usr/bin/env bash
# Control test: run FastWAM-OFFICIAL release ckpt through OUR V3 eval pipeline.
# Goal: isolate whether the ~3-pt libero_10 gap (vs paper 95.2%) comes from the
# trained ckpt or from our eval pipeline.
#
# Re-uses run_eval_libero_wanfastwam.sh (HPC3 layout) with local path overrides.
# This server (`server_wanfastwam.py`) key-maps FastWAM weights into a starVLA
# framework AND already installs state stats — so it's a fair comparison to our
# fwalign V3 eval.
#
# Defaults: only libero_10 (the suite where the gap lives) for speed.
# Override SUITES env to run all four.

set -u

STARVLA_DIR=${STARVLA_DIR:-/data/LFT-W02_data/junjie/VLA_WM/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/data/LFT-W02_data/junjie/LIBERO}
STARVLA_PYTHON=${STARVLA_PYTHON:-/data/LFT-W02_data/.conda/envs/starVLA/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/data/LFT-W02_data/.conda/envs/libero/bin/python}

# FastWAM-aligned config + official release ckpt
CONFIG_YAML=${CONFIG_YAML:-${STARVLA_DIR}/examples/LIBERO/train_files/starvla_wanfastwam_libero_fwalign_local.yaml}
FASTWAM_CKPT=${FASTWAM_CKPT:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM/checkpoints/fastwam_release/fastwam/libero_uncond_2cam224.pt}
WAN22_PATH=${WAN22_PATH:-/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers}
# Use our fwalign run dir's dataset_statistics.json — has the {"franka": {"state": ...}}
# schema that server_wanfastwam.py expects for set_state_stats. FastWAM-release's
# json uses a different flat schema and triggers "no proprio entry; skipping state norm",
# which silently kills SR (same root cause as V1 fwalign eval — see memory).
DATASET_STATS=${DATASET_STATS:-/data/LFT-W02_data/junjie/VLA_WM/fwalign_olabots_ckpts/dataset_statistics.json}

# Wan2_fastwam env-var escape hatches (DiffSynth-Studio safetensors).
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-${STARVLA_FASTWAM_REPO_PATH}/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

# Use NEW ports so we don't collide with the in-flight V3 run on 6694/6695.
PORTS=${PORTS:-"7694 7695"}
GPUS=${GPUS:-"0 1"}
N_TRIALS=${N_TRIALS:-50}
N_WORKERS_PER_SUITE=${N_WORKERS_PER_SUITE:-5}
TASKS_PER_WORKER=${TASKS_PER_WORKER:-2}

# Default: libero_10 only — quickest path to the diagnostic answer.
SUITES=${SUITES:-"libero_10"}

# Eval defaults already FastWAM-aligned via code defaults:
#   model2libero_interface.py: REPLAN_STEPS=10, USE_FWAM_ENSEMBLER=0
#   eval_libero.py: num_steps_wait=30

# Sanity checks
for p in "${FASTWAM_CKPT}" "${WAN22_PATH}" "${DATASET_STATS}" "${CONFIG_YAML}" \
         "${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" \
         "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
echo "[ok] FASTWAM_CKPT=${FASTWAM_CKPT}"
echo "[ok] WAN22_PATH=${WAN22_PATH}"
echo "[ok] DATASET_STATS=${DATASET_STATS}"
echo "[ok] CONFIG_YAML=${CONFIG_YAML}"
echo "[ok] PORTS=${PORTS}  SUITES=${SUITES}"
echo

cd "${STARVLA_DIR}"
export PYTHONPATH=${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}
export LIBERO_HOME LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

CKPT_DIR=$(dirname "${FASTWAM_CKPT}")  # for log dir
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_fwofficial_ctrl}
RUN_DIR=${STARVLA_DIR}/playground/eval_logs/${RUN_TAG}
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
echo "[ok] RUN_DIR=${RUN_DIR}"

HOST=127.0.0.1
PORT_ARR=(${PORTS})
GPU_ARR=(${GPUS})
SUITE_ARR=(${SUITES})

start_server() {
  local gpu="$1" port="$2"
  echo "[$(date +%H:%M:%S)] server gpu=${gpu} port=${port}"
  CUDA_VISIBLE_DEVICES=${gpu} ${STARVLA_PYTHON} deployment/model_server/server_wanfastwam.py \
      --config_yaml "${CONFIG_YAML}" \
      --fastwam_ckpt "${FASTWAM_CKPT}" \
      --wan22_path "${WAN22_PATH}" \
      --dataset_stats "${DATASET_STATS}" \
      --port ${port} \
      --use_bf16 \
      --idle_timeout -1 \
      > "${LOG_DIR}/server_gpu${gpu}_p${port}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port="$1"
  for i in $(seq 1 1500); do
    if (echo > /dev/tcp/${HOST}/${port}) 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] server :${port} READY"; return 0
    fi
    sleep 1
  done
  echo "[$(date +%H:%M:%S)] server :${port} FAILED to come up"; return 1
}

declare -a SERVER_PIDS=()
for i in "${!PORT_ARR[@]}"; do
  pid=$(start_server "${GPU_ARR[$i]}" "${PORT_ARR[$i]}")
  SERVER_PIDS+=("$pid")
done
cleanup() { for p in "${SERVER_PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

for p in "${PORT_ARR[@]}"; do wait_for_server "$p" || exit 1; done

declare -a CLIENT_PIDS=()
RR=0
for suite in "${SUITE_ARR[@]}"; do
  for w in $(seq 0 $((N_WORKERS_PER_SUITE - 1))); do
    s=$((w * TASKS_PER_WORKER))
    e=$((s + TASKS_PER_WORKER))
    port=${PORT_ARR[$((RR % ${#PORT_ARR[@]}))]}
    RR=$((RR + 1))
    seed=$((42 + w * 1000 + RANDOM % 100))
    log="${LOG_DIR}/eval_${suite}_w${w}.log"
    echo "[$(date +%H:%M:%S)] ${suite} w${w} tasks[${s},${e}) :${port} seed ${seed}"
    (
      # Client reads config.yaml + dataset_statistics.json from --pretrained-path's
      # run dir. FastWAM release ckpt has no such dir, so point client at our
      # fwalign run dir (same dataset stats source). Server still loads FW ckpt.
      ${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
          --args.pretrained-path "/data/LFT-W02_data/junjie/VLA_WM/fwalign_olabots_ckpts/final_model/pytorch_model.pt" \
          --args.host ${HOST} \
          --args.port ${port} \
          --args.task-suite-name ${suite} \
          --args.num-trials-per-task ${N_TRIALS} \
          --args.task-id-start ${s} \
          --args.task-id-end ${e} \
          --args.worker-id ${w} \
          --args.seed ${seed} \
          --args.video-out-path "${RUN_DIR}/results/${suite}/w${w}" \
          > "${log}" 2>&1
      echo "[$(date +%H:%M:%S)] DONE ${suite} w${w} rc=$?"
    ) &
    CLIENT_PIDS+=($!)
  done
done

echo "Total clients: ${#CLIENT_PIDS[@]}"
for pid in "${CLIENT_PIDS[@]}"; do wait "$pid"; done
echo "[$(date +%H:%M:%S)] all clients done"

echo "=== AGGREGATE ==="
for suite in "${SUITE_ARR[@]}"; do
  out_root="${RUN_DIR}/results/${suite}"
  ${STARVLA_PYTHON} - "$out_root" "$suite" <<'PY'
import json, sys, glob, os
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
echo "=== DONE — control test results in ${RUN_DIR} ==="
