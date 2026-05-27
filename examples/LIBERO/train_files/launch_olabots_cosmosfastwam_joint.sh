#!/usr/bin/env bash
# Launch Cosmos FastWAM joint training on Olabots (single-node, 8x H100 80GB).
#
# Sibling of launch_olabots_fwalign_joint.sh. Same paths/env contract but
# routes to the Cosmos-Predict2-2B-Video2World backbone + CosmosVideoDiT
# (FastWAM-style block layout) via use_fastwam_aligned_io=true on a
# cosmos-predict2 model id.
#
# Status: training pipeline verified end-to-end on LFT-W02 (synthetic batch
# forward + backward + optimizer step). The diffusers state-dict → CosmosVideoDiT
# remap copies attn/ffn/head linear weights but NOT modulation parameters
# (AdaLN-Lora → FastWAM modulation isn't a clean remap). Expect first-N-step
# loss to be higher than Wan FastWAM because modulation has to be re-learned.
#
# ssh olabots
# tmux new -s cosmosjoint
# WANDB_API_KEY=<key> bash examples/LIBERO/train_files/launch_olabots_cosmosfastwam_joint.sh

set -euo pipefail

# === Olabots layout ===
export STARVLA_DIR=${STARVLA_DIR:-/data/users/junjie/starVLA}
# FastWAM source — only needed for the rope_apply / mot interop paths.
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/users/junjie/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-/data/shared/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

# === Cosmos model + LIBERO data paths ===
# Cosmos-Predict2-2B-Video2World should be available locally. The yaml /
# launcher CLI sets framework.world_model.base_wm to this path.
export BASE_WM=${BASE_WM:-/data/shared/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World}
export LIBERO_DATA=${LIBERO_DATA:-/data/shared/datasets/libero_fastwam}

# No Cosmos-specific text cache yet (T5 dim 1024 != UMT5 4096; Wan cache
# can't be reused). Leave unset — T5 runs live, ~5GB extra VRAM per GPU.
export TEXT_EMBED_CACHE=${TEXT_EMBED_CACHE:-}

# No Cosmos VAE stats yet — Wan-VAE stats can't be reused (different z-dist).
# Leave unset — dataloader uses running estimate.
export FASTWAM_STATS=${FASTWAM_STATS:-}

# Conda env
export ACCELERATE_BIN=${ACCELERATE_BIN:-/data/users/wchen/envs/starvla/py310/bin/accelerate}

# GPU / batch sizing — Cosmos 2B is smaller than Wan 5B so we can match the
# fwalign joint EFF_BS=128 with BS=8, GA=2 even with the extra A→V joint
# attention overhead.
export NUM_GPUS=${NUM_GPUS:-8}
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-2}
export MAX_STEPS=${MAX_STEPS:-21700}
export WARMUP_STEPS=${WARMUP_STEPS:-1085}
export VIDEO_BACKEND=${VIDEO_BACKEND:-torchvision_av}

# wandb fallback
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[warn] WANDB_API_KEY not set — switching wandb to offline."
    export WANDB_MODE=offline
    export WANDB_API_KEY=offline_no_key_set
fi

# Sanity
for p in "$BASE_WM" "$LIBERO_DATA" "$STARVLA_FASTWAM_REPO_PATH/src/fastwam"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
echo "[ok] all required paths exist (Cosmos joint training)"
echo "  STARVLA_DIR:           $STARVLA_DIR"
echo "  BASE_WM (Cosmos):      $BASE_WM"
echo "  LIBERO_DATA:           $LIBERO_DATA"
echo "  FastWAM repo:          $STARVLA_FASTWAM_REPO_PATH"
echo "  ACCELERATE:            $ACCELERATE_BIN"
echo "  TEXT cache:            ${TEXT_EMBED_CACHE:-<live T5>}"
echo "  VAE stats:             ${FASTWAM_STATS:-<running estimate>}"
echo "  NUM_GPUS=$NUM_GPUS  BATCH=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  EFF_BS=$((NUM_GPUS*BATCH_SIZE*GRAD_ACCUM))"
echo "  mot_attention_mode:    joint (yaml + CLI override)"
echo

cd "$STARVLA_DIR"
exec bash examples/LIBERO/train_files/run_libero_train_cosmosfastwam_joint.sh "$@"
