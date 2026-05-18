#!/bin/bash
# LIBERO eval for fwalign final_model trained on Olabots (BS=16, MAX=21700).
# Uses starVLA's native server_policy.py + baseframework.from_pretrained, same
# pipeline as v3par3-repro eval — only the CKPT changes.
#
# Prerequisites:
#   1. Rsync'd run dir at FWALIGN_RUN_DIR (default below) — must contain
#      final_model/pytorch_model.pt + config.yaml + dataset_statistics.json
#   2. config.yaml paths must point at LOCAL files (already patched; backup at
#      config.yaml.olabots.bak).
#   3. Env vars below tell Wan2_fastwam where DiffSynth-Studio safetensors live.
#
# Usage:
#   bash examples/LIBERO/eval_files/run_eval_fwalign_local.sh                  # all 4 suites
#   SUITES="libero_10" bash examples/LIBERO/eval_files/run_eval_fwalign_local.sh
set -u

FWALIGN_RUN_DIR=${FWALIGN_RUN_DIR:-/data/LFT-W02_data/junjie/VLA_WM/fwalign_olabots_ckpts}
export CKPT=${CKPT:-${FWALIGN_RUN_DIR}/final_model/pytorch_model.pt}

# Wan2_fastwam env-var escape hatches (preferred over yaml on path mismatch).
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-${STARVLA_FASTWAM_REPO_PATH}/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

# Eval defaults (v3par3-aligned). Override via env.
export N_TRIALS=${N_TRIALS:-50}
export SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
export BATCH_SIZE=${BATCH_SIZE:-10}
export BATCH_WAIT_MS=${BATCH_WAIT_MS:-20}

# Sanity checks
for p in "${CKPT}" "${FWALIGN_RUN_DIR}/config.yaml" "${FWALIGN_RUN_DIR}/dataset_statistics.json" \
         "${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" \
         "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
echo "[ok] CKPT=${CKPT}"
echo "[ok] STARVLA_FASTWAM_REPO=${STARVLA_FASTWAM_REPO_PATH}"
echo "[ok] STARVLA_FASTWAM_CKPTS=${STARVLA_FASTWAM_CHECKPOINTS_ROOT}"
echo "[ok] SUITES=${SUITES}  N_TRIALS=${N_TRIALS}"

# Delegate to the fwalign-specific parallel eval (uses server_wanfastwam_starvla.py
# which installs state stats from dataset_statistics.json).
exec bash "$(dirname "$0")/eval_all_parallel_fwalign.sh" "$@"
