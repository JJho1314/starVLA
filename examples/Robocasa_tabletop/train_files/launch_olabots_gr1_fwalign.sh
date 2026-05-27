#!/usr/bin/env bash
# Olabots wrapper for Robocasa Fourier-GR1 fwalign (non-joint) training.
# Sibling of launch_olabots_fwalign.sh / launch_olabots_fwalign_joint.sh.
# Sets olabots-specific paths and calls the generic GR1 launcher.
#
# Recommended:
#   ssh olabots && tmux new -s gr1fwalign
#   bash examples/Robocasa_tabletop/train_files/launch_olabots_gr1_fwalign.sh

set -euo pipefail

# === Olabots layout ===
export STARVLA_DIR=${STARVLA_DIR:-/data/users/junjie/starVLA}
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/users/junjie/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-/data/shared/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

# === Wan backbone weights ===
export BASE_WM=${BASE_WM:-/data/shared/checkpoints/Wan-AI/Wan2.2-TI2V-5B-Diffusers}

# === GR1 dataset (PhysicalAI-Robotics-GR00T-X-Embodiment-Sim on olabots) ===
export GR1_DATA=${GR1_DATA:-/data/shared/datasets/huggingface/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}

# === Conda env ===
export ACCELERATE_BIN=${ACCELERATE_BIN:-/data/users/wchen/envs/starvla/py310/bin/accelerate}

# === GPU / batch sizing ===
# Same EFF_BS=128 as fwalign LIBERO 97.05 recipe. GR1 video 224×224 single-cam
# uses less mem than LIBERO 224×448 two-cam, so BS=8 GA=2 should fit on H100 80GB
# (vs Cosmos joint which needed BS=4 GA=4 because action-dit hidden=2048 doubled).
export NUM_GPUS=${NUM_GPUS:-8}
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-2}
export MAX_STEPS=${MAX_STEPS:-80000}
export WARMUP_STEPS=${WARMUP_STEPS:-4000}
export VIDEO_BACKEND=${VIDEO_BACKEND:-torchvision_av}

# === WandB (cloud) ===
export WANDB_MODE=${WANDB_MODE:-online}
# Force public wandb.ai, override any private-server default from inner launcher
# or ~/.bashrc (olabots .bashrc sets api.bandw.top; inner script defaults to
# the HPC3 internal 10.12.1.245:8080).
export WANDB_BASE_URL=${WANDB_BASE_URL_OVERRIDE:-https://api.wandb.ai}
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set}
export WANDB_ENTITY=${WANDB_ENTITY:-jjho1314}
export WANDB_PROJECT=${WANDB_PROJECT:-starVLA_Robocasa}

# === Sanity ===
for p in "$BASE_WM" "$GR1_DATA" "$STARVLA_FASTWAM_REPO_PATH/src/fastwam" \
         "$STARVLA_FASTWAM_CHECKPOINTS_ROOT/DiffSynth-Studio"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
echo "[ok] all required paths exist (GR1 fwalign training)"
echo "  BASE_WM:               $BASE_WM"
echo "  GR1_DATA:              $GR1_DATA"
echo "  FastWAM repo:          $STARVLA_FASTWAM_REPO_PATH"
echo "  ACCELERATE:            $ACCELERATE_BIN"
echo "  WANDB_MODE:            $WANDB_MODE  (key=${WANDB_API_KEY:0:6}…)"
echo "  WANDB_PROJECT/ENTITY:  $WANDB_PROJECT / $WANDB_ENTITY"
echo "  NUM_GPUS=$NUM_GPUS  BATCH=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  EFF_BS=$((NUM_GPUS*BATCH_SIZE*GRAD_ACCUM))"
echo

cd "$STARVLA_DIR"
exec bash examples/Robocasa_tabletop/train_files/run_robocasa_train_wanfastwam_fwalign_gr1.sh "$@"
