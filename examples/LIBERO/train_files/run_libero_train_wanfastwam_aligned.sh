#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# WanFastWAM (FastWAM-aligned) LIBERO training on starVLA.
#
# Reproduces FastWAM's training setup:
#   - Joint MoT video+action forward (single backbone pass)
#   - Gaussian training_weight loss reweighting
#   - First-frame clean latent replacement (TI2V)
#   - Proprio appended to text context
#   - shift=5.0 for both video and action schedulers
#   - lambda_video=1.0, lambda_action=1.0
#
# Usage:
#   # 4-GPU (default)
#   bash examples/LIBERO/train_files/run_libero_train_wanfastwam_aligned.sh
#
#   # 8-GPU
#   NUM_GPUS=8 bash examples/LIBERO/train_files/run_libero_train_wanfastwam_aligned.sh
#
#   # Resume from checkpoint
#   RESUME_STEP=40000 bash examples/LIBERO/train_files/run_libero_train_wanfastwam_aligned.sh
#
# Environment variables:
#   NUM_GPUS          Number of GPUs (default: 4)
#   BASE_WM           Path to Wan2.2-TI2V-5B-Diffusers weights
#   LIBERO_DATA       Path to LIBERO lerobot datasets
#   BATCH_SIZE        Per-device batch size (default: 4)
#   GRAD_ACCUM        Gradient accumulation steps (default: 4)
#   MAX_STEPS         Max training steps (default: 80000)
#   RESUME_STEP       Step to resume from (default: none)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export MASTER_PORT=$((29500 + RANDOM % 1000))

# ─── Wandb config ─────────────────────────────────────────────────────────────
# HPC3 has a local wandb server at 10.12.1.245:8080 (used by siiRL etc).
# Default to "online" against that server. Override:
#   WANDB_MODE=offline                   → write locally; sync later via `wandb sync`
#   WANDB_MODE=disabled                  → no wandb at all
#   WANDB_BASE_URL=https://api.wandb.ai  → public wandb (only works if compute node has external internet)
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.12.1.245:8080}"
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set (export it in your env, e.g. ~/.bashrc)}"
export WANDB_ENTITY="${WANDB_ENTITY:-jjho1314}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Libero}"

# ─── Configurable paths ──────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
BASE_WM="${BASE_WM:-/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers}"
LIBERO_DATA="${LIBERO_DATA:-/data/LFT-W02_data/junjie/data/libero_datasets/libero_lerobot}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
# Pick exactly ONE budget — NUM_EPOCHS (FastWAM-style, derives steps from dataloader)
# OR MAX_STEPS (explicit). NUM_EPOCHS overrides MAX_STEPS when set.
NUM_EPOCHS="${NUM_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-80000}"
RESUME_STEP="${RESUME_STEP:-}"

Framework_name=WanFastWAM
config_yaml=./examples/LIBERO/train_files/starvla_wanfastwam_libero.yaml
data_mix=libero_all_fastwam
run_root_dir=./playground/Checkpoints
run_id=$(date +%Y%m%d_%H%M%S)_libero_${Framework_name}_aligned

# ─── Effective batch size info ────────────────────────────────────────────────
EFF_BS=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))
if [[ -n "${NUM_EPOCHS}" ]]; then
    BUDGET_DESC="num_epochs=${NUM_EPOCHS} (max_steps will be derived)"
else
    BUDGET_DESC="max_steps=${MAX_STEPS}"
fi
echo "============================================================"
echo "WanFastWAM (FastWAM-aligned) LIBERO Training"
echo "============================================================"
echo "GPUs:                ${NUM_GPUS}"
echo "Per-device BS:       ${BATCH_SIZE}"
echo "Grad accum:          ${GRAD_ACCUM}"
echo "Effective BS:        ${EFF_BS}"
echo "Budget:              ${BUDGET_DESC}"
echo "Base WM:             ${BASE_WM}"
echo "Data:                ${LIBERO_DATA}"
echo "Run ID:              ${run_id}"
echo "============================================================"

# ─── Create output dir & copy script ─────────────────────────────────────────
output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# ─── Resume args ──────────────────────────────────────────────────────────────
RESUME_ARGS=""
if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_ARGS="--trainer.is_resume true --trainer.resume_step ${RESUME_STEP}"
    echo "Resuming from step ${RESUME_STEP}"
fi

# ─── Budget args ──────────────────────────────────────────────────────────────
# When NUM_EPOCHS is set, only pass --trainer.num_epochs; the trainer derives
# max_train_steps after the dataloader is built (see _derive_max_steps_from_epochs).
# Otherwise pass --trainer.max_train_steps explicitly.
if [[ -n "${NUM_EPOCHS}" ]]; then
    BUDGET_ARGS="--trainer.num_epochs ${NUM_EPOCHS}"
else
    BUDGET_ARGS="--trainer.max_train_steps ${MAX_STEPS}"
fi

# ─── Launch training ─────────────────────────────────────────────────────────
# Optional pre-computed UMT5 text embeds cache. When set, Wan2.py skips loading
# UMT5-XXL on each GPU (~11 GB saved per rank → enables larger per-device bs).
TEXT_EMBED_ARGS=""
if [[ -n "${TEXT_EMBED_CACHE:-}" ]]; then
    TEXT_EMBED_ARGS="--framework.world_model.text_embed_cache_path ${TEXT_EMBED_CACHE}"
    echo "Text embeds:        cached @ ${TEXT_EMBED_CACHE}"
fi

accelerate launch \
    --config_file "${DS_CONFIG_OVERRIDE:-starVLA/config/deepseeds/deepspeed_zero2_fastwam.yaml}" \
    --num_processes "${NUM_GPUS}" \
    --main_process_port "${MASTER_PORT}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name "${Framework_name}" \
    --framework.world_model.base_wm "${BASE_WM}" \
    --framework.qwenvl.base_vlm "${BASE_WM}" \
    --framework.action_dit.skip_pretrained_load false \
    ${TEXT_EMBED_ARGS} \
    --framework.fastwam.lambda_video 1.0 \
    --framework.fastwam.lambda_action 1.0 \
    --framework.fastwam.enable_video_loss true \
    --datasets.vla_data.data_root_dir "${LIBERO_DATA}" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
    ${BUDGET_ARGS} \
    --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --trainer.save_interval 10000 \
    --trainer.logging_frequency 100 \
    --trainer.eval_interval 100 \
    --trainer.enable_mixed_precision_training true \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project starVLA_Libero \
    --wandb_entity jjho1314 \
    ${RESUME_ARGS} \
    "$@"
