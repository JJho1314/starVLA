#!/usr/bin/env bash
# Launch fwalign training on Olabots (single-node, 8x H100 80GB).
# No SLURM — runs in foreground / tmux. Wrap the generic
# run_libero_train_wanfastwam_fwalign.sh with Olabots-specific paths.
#
# Recommended usage (so you don't lose the run if SSH disconnects):
#   ssh olabots
#   tmux new -s fwalign
#   bash examples/LIBERO/train_files/launch_olabots_fwalign.sh
#   # Ctrl+B D to detach. tmux attach -t fwalign to reconnect.
#
# Override knobs (env vars):
#   NUM_GPUS=4               — use fewer GPUs (default: all 8)
#   BATCH_SIZE=8 GRAD_ACCUM=2 — effective batch knobs (default 8x2x8 = 128)
#   MAX_STEPS=21700          — total training steps
#   WANDB_API_KEY=... export before calling, or set in ~/.bashrc

set -euo pipefail

# === Olabots layout ===
export STARVLA_DIR=${STARVLA_DIR:-/data/users/junjie/starVLA}
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/users/junjie/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-/data/shared/checkpoints}
# Belt-and-suspenders: bypass yaml's CLI override fragility by pinning the
# DiffSynth loader path via env var. setdefault in Wan2_fastwam.py respects this.
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

# Tell the generic launcher where to find things on this machine.
export BASE_WM=${BASE_WM:-/data/shared/checkpoints/Wan-AI/Wan2.2-TI2V-5B-Diffusers}
export LIBERO_DATA=${LIBERO_DATA:-/data/shared/datasets/libero_fastwam}
export TEXT_EMBED_CACHE=${TEXT_EMBED_CACHE:-/data/shared/checkpoints/fastwam_text_cache_libero}
export FASTWAM_STATS=${FASTWAM_STATS:-/data/shared/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}

# === Olabots conda env (3.10, torch 2.6, owned by us despite the /wchen/ path) ===
export ACCELERATE_BIN=${ACCELERATE_BIN:-/data/users/wchen/envs/starvla/py310/bin/accelerate}

# === GPU / batch sizing ===
export NUM_GPUS=${NUM_GPUS:-8}
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-2}
export MAX_STEPS=${MAX_STEPS:-21700}
export WARMUP_STEPS=${WARMUP_STEPS:-1085}
export VIDEO_BACKEND=${VIDEO_BACKEND:-torchvision_av}

# === wandb defaults (offline if no key — avoid the launcher's strict assertion) ===
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[warn] WANDB_API_KEY not set — switching wandb to offline."
    export WANDB_MODE=offline
    # The generic launcher uses ${WANDB_API_KEY:?...} which would abort; set a
    # dummy value so the assertion passes (offline mode ignores the value).
    export WANDB_API_KEY=offline_no_key_set
fi

# === Sanity checks for Olabots-specific paths ===
for p in "$BASE_WM" "$LIBERO_DATA" "$TEXT_EMBED_CACHE" "$STARVLA_FASTWAM_REPO_PATH/src/fastwam" \
         "$STARVLA_FASTWAM_CHECKPOINTS_ROOT/DiffSynth-Studio"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
echo "[ok] all required paths exist"
echo "  BASE_WM:               $BASE_WM"
echo "  LIBERO_DATA:           $LIBERO_DATA"
echo "  TEXT_EMBED_CACHE:      $TEXT_EMBED_CACHE"
echo "  FASTWAM_STATS:         $FASTWAM_STATS"
echo "  STARVLA_FASTWAM_REPO:  $STARVLA_FASTWAM_REPO_PATH"
echo "  STARVLA_FASTWAM_CKPTS: $STARVLA_FASTWAM_CHECKPOINTS_ROOT"
echo "  ACCELERATE:            $ACCELERATE_BIN"
echo "  NUM_GPUS=$NUM_GPUS  BATCH=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  EFF_BS=$((NUM_GPUS*BATCH_SIZE*GRAD_ACCUM))"
echo

cd "$STARVLA_DIR"
exec bash examples/LIBERO/train_files/run_libero_train_wanfastwam_fwalign.sh "$@"
