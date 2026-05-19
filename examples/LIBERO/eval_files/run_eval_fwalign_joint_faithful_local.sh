#!/usr/bin/env bash
# LIBERO eval — joint-FAITHFUL inference (separate from the standard fwalign
# and the cache-reuse "approximate-joint" launchers; this one does NOT touch
# `eval_all_parallel_fwalign.sh` or any of the V8 fwalign pipeline files).
#
# Path used here:
#   server  : deployment/model_server/server_wanfastwam_starvla_joint_faithful.py
#             (rebinds predict_action → predict_action_joint)
#   loop    : eval_all_parallel_fwalign_joint_faithful.sh
#   model   : WanFastWAM.predict_action_joint
#             (per-step full MoT forward, no KV cache reuse, video+action
#              denoised in lockstep with first-frame pinned to clean latent)
#
# Matches `FastWAMJoint.infer_action` upstream
# (fastwam/models/wan22/fastwam_joint.py:96). Use this for ckpts that were
# trained with `framework.fastwam.mot_attention_mode: joint` AND when you need
# strict numerical parity with upstream FastWAMJoint inference.
#
# If you only need approximate joint inference (mask flipped to joint but video
# K/V still reused across action denoise steps), use `run_eval_fwalign_joint_local.sh`
# instead — it's significantly cheaper, just less faithful to the training-time
# joint denoise schedule.
#
# Usage:
#   bash examples/LIBERO/eval_files/run_eval_fwalign_joint_faithful_local.sh
#   SUITES="libero_10" bash examples/LIBERO/eval_files/run_eval_fwalign_joint_faithful_local.sh
#   FWALIGN_RUN_DIR=/path/to/joint_trained_ckpt \
#     bash examples/LIBERO/eval_files/run_eval_fwalign_joint_faithful_local.sh

set -u

FWALIGN_RUN_DIR=${FWALIGN_RUN_DIR:-/data/LFT-W02_data/junjie/VLA_WM/fwalign_olabots_ckpts}
export CKPT=${CKPT:-${FWALIGN_RUN_DIR}/final_model/pytorch_model.pt}

# Wan2_fastwam env-var escape hatches (DiffSynth-Studio safetensors).
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-${STARVLA_FASTWAM_REPO_PATH}/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

# Eval defaults (V8-aligned via code defaults).
export N_TRIALS=${N_TRIALS:-50}
export SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
# Joint-faithful inference is much more expensive per request (≈NSW× the
# standard path). Keep per-server batch_size conservative.
export BATCH_SIZE=${BATCH_SIZE:-5}
export BATCH_WAIT_MS=${BATCH_WAIT_MS:-20}

# Warn if config.yaml didn't request joint mode (the framework would otherwise
# default to fastwam mask, which would silently use the wrong attention pattern
# for joint-trained ckpts).
if [[ -f "${FWALIGN_RUN_DIR}/config.yaml" ]]; then
    if ! grep -q "mot_attention_mode:[[:space:]]*joint" "${FWALIGN_RUN_DIR}/config.yaml"; then
        echo "[warn] ${FWALIGN_RUN_DIR}/config.yaml does NOT set mot_attention_mode: joint" >&2
        echo "       → framework will load with mot_attention_mode='fastwam'." >&2
        echo "       The server-side predict_action_joint will still run the joint A→V mask," >&2
        echo "       but if the ckpt was NOT trained with joint mask, SR will degrade." >&2
        echo "       To force joint mode, edit config.yaml: framework.fastwam.mot_attention_mode: joint" >&2
    fi
fi

for p in "${CKPT}" "${FWALIGN_RUN_DIR}/config.yaml" "${FWALIGN_RUN_DIR}/dataset_statistics.json" \
         "${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" \
         "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
echo "[ok] CKPT=${CKPT}"
echo "[ok] FWALIGN_RUN_DIR=${FWALIGN_RUN_DIR}"
echo "[ok] inference path: joint_faithful (predict_action_joint, per-step full MoT)"
echo "[ok] SUITES=${SUITES}  N_TRIALS=${N_TRIALS}  BATCH_SIZE=${BATCH_SIZE}"

exec bash "$(dirname "$0")/eval_all_parallel_fwalign_joint_faithful.sh" "$@"
