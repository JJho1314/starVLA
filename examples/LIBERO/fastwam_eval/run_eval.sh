#!/bin/bash
# One-click launcher for FastWAM-style LIBERO evaluation, run from starVLA repo root.
#
# This is verbatim FastWAM's eval pipeline (run_libero_manager.py + run_libero_parallel_test.sh
# + eval_libero_single.py) copied into starVLA, with paths adjusted to live under
# examples/LIBERO/fastwam_eval/.
#
# Requirements:
#   1) `fastwam` conda env with the FastWAM package installed (editable from this checkout
#      or any other; we rely on `import fastwam`).
#   2) LIBERO sim (mujoco==3.3.2 per FastWAM README) installed in that env.
#   3) Wan2.2-TI2V-5B weights at `./checkpoints/Wan-AI/Wan2.2-TI2V-5B` (FastWAM-style layout).
#   4) ActionDiT preprocessed file at `./checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
#      (run `scripts/preprocess_action_dit_backbone.py` from the FastWAM repo to generate).
#   5) Released ckpt + dataset stats at `./checkpoints/fastwam_release/`.
#
# Defaults below match LFT-W02 layout. Override via env vars.
#
# Usage:
#   bash examples/LIBERO/fastwam_eval/run_eval.sh
#
# Tunable env:
#   FASTWAM_PYTHON=...          # python in fastwam env
#   FASTWAM_REPO=...            # FastWAM repo root (for sys.path); only needed if `import fastwam` fails
#   CKPT=...                    # release ckpt path (default: ./checkpoints/fastwam_release/libero_uncond_2cam224.pt)
#   STATS=...                   # release dataset stats path
#   NUM_GPUS=1
#   NUM_TRIALS=50
#   MAX_TASKS_PER_GPU=1
#   TASK=libero_uncond_2cam224_1e-4
#   SUITES="libero_spatial libero_object libero_goal libero_10"

set -u

STARVLA_DIR=${STARVLA_DIR:-$(cd "$(dirname "$0")"/../../.. && pwd)}
FASTWAM_PYTHON=${FASTWAM_PYTHON:-/data/LFT-W02_data/.conda/envs/fastwam/bin/python}
FASTWAM_REPO=${FASTWAM_REPO:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM}
LIBERO_HOME=${LIBERO_HOME:-/data/LFT-W02_data/junjie/LIBERO}

CKPT=${CKPT:-${STARVLA_DIR}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}
STATS=${STATS:-${STARVLA_DIR}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}

TASK=${TASK:-libero_uncond_2cam224_1e-4}
NUM_GPUS=${NUM_GPUS:-1}
NUM_TRIALS=${NUM_TRIALS:-50}
MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU:-1}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

cd "${STARVLA_DIR}"

# Make `fastwam` importable (env may already have it editable; this is a belt-and-suspenders).
export PYTHONPATH="${FASTWAM_REPO}/src:${LIBERO_HOME}:${PYTHONPATH:-}"
export LIBERO_HOME LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
# FastWAM expects raw Wan2.2 weights under ./checkpoints/Wan-AI/Wan2.2-TI2V-5B
export DIFFSYNTH_MODEL_BASE_PATH="${STARVLA_DIR}/checkpoints"
export EVAL_PYTHON="${FASTWAM_PYTHON}"   # parallel_test.sh dispatches subprocesses via this

if [[ ! -f "${CKPT}" ]]; then
  echo "[fatal] release ckpt not found: ${CKPT}" >&2; exit 1
fi
if [[ ! -e "${STARVLA_DIR}/checkpoints/Wan-AI/Wan2.2-TI2V-5B" ]]; then
  echo "[fatal] missing Wan2.2 weights symlink at ./checkpoints/Wan-AI/Wan2.2-TI2V-5B" >&2; exit 1
fi
if [[ ! -f "${STARVLA_DIR}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" ]]; then
  echo "[fatal] missing ActionDiT preprocessed file at ./checkpoints/" >&2; exit 1
fi

OUT_BASE=${OUT_BASE:-${STARVLA_DIR}/playground/eval_logs/$(date +%Y%m%d_%H%M%S)_fastwam_eval}
mkdir -p "${OUT_BASE}"

for suite in ${SUITES}; do
  echo ""
  echo "=== eval suite: ${suite} ==="
  ${FASTWAM_PYTHON} examples/LIBERO/fastwam_eval/run_libero_manager.py \
    task=${TASK} \
    ckpt="${CKPT}" \
    EVALUATION.task_suite_name=${suite} \
    EVALUATION.num_trials=${NUM_TRIALS} \
    EVALUATION.dataset_stats_path="${STATS}" \
    EVALUATION.output_dir="${OUT_BASE}/${suite}" \
    EVALUATION.action_horizon=${ACTION_HORIZON:-32} \
    EVALUATION.replan_steps=${REPLAN_STEPS:-5} \
    MULTIRUN.num_gpus=${NUM_GPUS} \
    MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU} 2>&1 | tee "${OUT_BASE}/${suite}_manager.log"
done

echo ""
echo "All suites done. Logs + per-suite results under: ${OUT_BASE}"
