#!/usr/bin/env bash
# LIBERO eval entrypoint for starVLA-native ckpts trained with the ORIGINAL
# `starVLA/model/modules/world_model/Wan2.py` (HF Diffusers/Transformers VAE+T5
# path). This is the v3par3-style setup, NOT the fwalign FastWAM-aligned IO one.
#
# Inherits ALL 5 V8 eval-pipeline fixes (see FASTWAM_EVAL_ALIGNMENT.md):
#   1. State stats install     — server_wanfastwam_starvla.py (best-effort:
#                                 installs only if dataset_statistics.json exists
#                                 with a state/proprio block; else passthrough,
#                                 which is fine for HF-preprocessing-trained
#                                 ckpts that don't expect min/max state norm).
#   2. num_steps_wait=30       — eval_libero.py default
#   3. replan_steps=10         — model2libero_interface.py default
#   4. use_action_ensembler=0  — model2libero_interface.py default
#   5. PIL BILINEAR resize     — model2libero_interface.py _resize_image
#   6. CPU device action noise — WanFastWAM.py predict_action
#
# Required:
#   STARVLA_RUN_DIR — run directory containing config.yaml + dataset_statistics.json
#                     + checkpoints/<name>.pt or final_model/pytorch_model.pt
#
# Usage:
#   STARVLA_RUN_DIR=/path/to/your/v3par3_run_dir \
#   bash examples/LIBERO/eval_files/run_eval_starvla_native_local.sh
#
#   # or just SUITES override:
#   SUITES="libero_10" STARVLA_RUN_DIR=... bash ...
set -u

STARVLA_RUN_DIR=${STARVLA_RUN_DIR:?STARVLA_RUN_DIR must be set to the run directory}
export CKPT=${CKPT:-${STARVLA_RUN_DIR}/final_model/pytorch_model.pt}

# If user pointed at a checkpoint .pt file instead of a run dir, derive run_dir.
if [[ -f "${STARVLA_RUN_DIR}" && "${STARVLA_RUN_DIR}" == *.pt ]]; then
    export CKPT="${STARVLA_RUN_DIR}"
    STARVLA_RUN_DIR=$(dirname "$(dirname "${CKPT}")")
fi

# Wan2.py uses HF Diffusers/Transformers — no fastwam vendored components needed.
# But we still set these env vars in case the run_dir's config.yaml references
# them (e.g., v3par3 yaml had text_embed_cache paths pointing to FastWAM dirs).
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-${STARVLA_FASTWAM_REPO_PATH}/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

# Eval defaults (V8-aligned via code defaults).
export N_TRIALS=${N_TRIALS:-50}
export SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
export BATCH_SIZE=${BATCH_SIZE:-10}
export BATCH_WAIT_MS=${BATCH_WAIT_MS:-20}

# Sanity checks
for p in "${CKPT}" "${STARVLA_RUN_DIR}/config.yaml"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
[[ -e "${STARVLA_RUN_DIR}/dataset_statistics.json" ]] || \
    echo "[warn] no dataset_statistics.json — state stats install will skip (OK for v3par3-style HF ckpts)"

echo "[ok] STARVLA_RUN_DIR=${STARVLA_RUN_DIR}"
echo "[ok] CKPT=${CKPT}"
echo "[ok] SUITES=${SUITES}  N_TRIALS=${N_TRIALS}"

# Delegate to fwalign parallel launcher — same server (server_wanfastwam_starvla.py)
# is used because (a) it correctly builds the framework via baseframework.from_pretrained
# which respects whichever world_model variant the config.yaml selects, and
# (b) _install_state_stats is best-effort (no-op if stats absent).
# We re-export FWALIGN_RUN_DIR because the inner script reads it.
export FWALIGN_RUN_DIR="${STARVLA_RUN_DIR}"
exec bash "$(dirname "$0")/eval_all_parallel_fwalign.sh" "$@"
