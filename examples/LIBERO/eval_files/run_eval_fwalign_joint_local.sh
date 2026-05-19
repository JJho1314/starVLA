#!/usr/bin/env bash
# LIBERO eval for fwalign ckpts trained with FastWAMJoint-style attention
# (`framework.fastwam.mot_attention_mode: joint`).
#
# Sibling of run_eval_fwalign_local.sh — same V8 pipeline (NSW=30, replan=10,
# ensembler=off, PIL BILINEAR, CPU unseeded RNG, mujoco 3.3.2), but the framework
# is built with `mot_attention_mode: joint` so action attends to ALL video tokens
# (matches FastWAMJoint upstream's mask).
#
# Requirements (in addition to run_eval_fwalign_local.sh's requirements):
#   - ckpt must have been trained with `mot_attention_mode: joint` (otherwise
#     the eval mask mismatches what the model learned, expect SR degradation).
#   - run dir's config.yaml should have `framework.fastwam.mot_attention_mode: joint`,
#     OR you point the launcher at one of our joint train_files yamls via
#     SERVER_CONFIG_YAML (see below).
#
# Limitation: inference still uses the prefill-then-cache video path (one
# clean-video forward; action denoising loop reuses K/V). This is APPROXIMATE
# joint inference: training noised the video latents, but here we feed clean.
# Matches what `_prefill_video_cache` does, just with the joint A→V mask. If
# you want fully faithful FW-joint inference (re-encode noisy video each step),
# extend WanFastWAM.predict_action with a joint-denoise branch (see
# fastwam_joint.FastWAMJoint.infer_action upstream for the reference loop).
#
# Usage:
#   bash examples/LIBERO/eval_files/run_eval_fwalign_joint_local.sh
#   SUITES="libero_10" bash examples/LIBERO/eval_files/run_eval_fwalign_joint_local.sh

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
export BATCH_SIZE=${BATCH_SIZE:-10}
export BATCH_WAIT_MS=${BATCH_WAIT_MS:-20}

# IMPORTANT: the server reads `mot_attention_mode` from the run dir's config.yaml.
# Make sure the ckpt's config.yaml has `framework.fastwam.mot_attention_mode: joint`.
# If the saved config doesn't have it, copy our joint train-files yaml on top:
#   cp examples/LIBERO/train_files/starvla_wanfastwam_libero_fwalign_joint_local.yaml \
#      $FWALIGN_RUN_DIR/config.yaml.joint
# and point STARVLA_RUN_DIR at a dir whose config.yaml has the flag.
#
# Quick check below to warn if the config doesn't request joint mode.
if [[ -f "${FWALIGN_RUN_DIR}/config.yaml" ]]; then
    if ! grep -q "mot_attention_mode:[[:space:]]*joint" "${FWALIGN_RUN_DIR}/config.yaml"; then
        echo "[warn] ${FWALIGN_RUN_DIR}/config.yaml does NOT set mot_attention_mode: joint" >&2
        echo "       → server will default to 'fastwam' mode. To use joint, edit the yaml or" >&2
        echo "       set: framework.fastwam.mot_attention_mode: joint" >&2
    fi
fi

# Sanity checks
for p in "${CKPT}" "${FWALIGN_RUN_DIR}/config.yaml" "${FWALIGN_RUN_DIR}/dataset_statistics.json" \
         "${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" \
         "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
echo "[ok] CKPT=${CKPT}"
echo "[ok] FWALIGN_RUN_DIR=${FWALIGN_RUN_DIR}"
echo "[ok] mot_attention_mode = joint (will surface in server boot log)"
echo "[ok] SUITES=${SUITES}  N_TRIALS=${N_TRIALS}"

exec bash "$(dirname "$0")/eval_all_parallel_fwalign.sh" "$@"
