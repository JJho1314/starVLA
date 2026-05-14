#!/usr/bin/env bash
# Reproduce the high-scoring 20260508 v3-par3 WanFastWAM LIBERO run.
#
# Verified against the original HPC3 run:
#   playground/Checkpoints/20260508_190938_libero_WanFastWAM_aligned/config.full.yaml
#
# Key differences from the generic aligned script:
#   - max_train_steps=21700, not 80000
#   - per_device_batch_size=8, gradient_accumulation_steps=2
#   - video_backend=torchvision_av, not torchcodec
#   - use the precomputed UMT5 text embedding cache
#
# Typical usage on HPC3:
#   bash examples/LIBERO/train_files/run_libero_train_wanfastwam_v3par3_repro.sh
#
# Useful overrides:
#   NUM_GPUS=8 bash examples/LIBERO/train_files/run_libero_train_wanfastwam_v3par3_repro.sh
#   WANDB_MODE=offline bash examples/LIBERO/train_files/run_libero_train_wanfastwam_v3par3_repro.sh

set -euo pipefail

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

TORCHVISION_VIDEO_WARNING_FILTER="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"
if [[ -z "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${TORCHVISION_VIDEO_WARNING_FILTER}"
else
    export PYTHONWARNINGS="${PYTHONWARNINGS},${TORCHVISION_VIDEO_WARNING_FILTER}"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.12.1.245:8080}"
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set (export it in your env, e.g. ~/.bashrc)}"
export WANDB_ENTITY="${WANDB_ENTITY:-jjho1314}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Libero}"

Framework_name=WanFastWAM
config_yaml="${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_wanfastwam_libero.yaml}"
data_mix="${DATA_MIX:-libero_all_fastwam}"
run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_libero_${Framework_name}_aligned_v3par3_repro}"

# Prefer the original HPC3 paths when present; otherwise fall back to the local LFT-W02 layout.
if [[ -d /data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers ]]; then
    default_base_wm=/data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers
else
    default_base_wm=/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers
fi

if [[ -d /data/user/jhe724/workspace/FastWAM/data/libero_mujoco3.3.2 ]]; then
    default_libero_data=/data/user/jhe724/workspace/FastWAM/data/libero_mujoco3.3.2
elif [[ -d /data/user/jhe724/workspace/data/libero_fastwam ]]; then
    default_libero_data=/data/user/jhe724/workspace/data/libero_fastwam
else
    default_libero_data=/data/LFT-W02_data/junjie/data/libero_datasets/libero_lerobot
fi

if [[ -d /data/user/jhe724/workspace/FastWAM/data/text_embeds_cache/libero ]]; then
    default_text_cache=/data/user/jhe724/workspace/FastWAM/data/text_embeds_cache/libero
elif [[ -f /data/user/jhe724/workspace/data/libero_fastwam/libero_umt5_text_embeds.pt ]]; then
    default_text_cache=/data/user/jhe724/workspace/data/libero_fastwam/libero_umt5_text_embeds.pt
else
    default_text_cache=/data/LFT-W02_data/junjie/data/libero_fastwam/libero_umt5_text_embeds.pt
fi

if [[ -f /data/user/jhe724/workspace/FastWAM/runs/libero_uncond_2cam224_1e-4/2026-05-09_16-19-41/dataset_stats.json ]]; then
    default_fastwam_stats=/data/user/jhe724/workspace/FastWAM/runs/libero_uncond_2cam224_1e-4/2026-05-09_16-19-41/dataset_stats.json
else
    default_fastwam_stats=""
fi

NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/user/jhe724/.conda/envs/starVLA/bin/accelerate}"
BASE_WM="${BASE_WM:-${default_base_wm}}"
LIBERO_DATA="${LIBERO_DATA:-${default_libero_data}}"
TEXT_EMBED_CACHE="${TEXT_EMBED_CACHE:-${default_text_cache}}"
FASTWAM_STATS="${FASTWAM_STATS:-${default_fastwam_stats}}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_STEPS="${MAX_STEPS:-21700}"
WARMUP_STEPS="${WARMUP_STEPS:-1085}"
RESUME_STEP="${RESUME_STEP:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"

if [[ ! -d "${BASE_WM}" ]]; then
    echo "Missing BASE_WM: ${BASE_WM}" >&2
    exit 1
fi
if [[ ! -d "${LIBERO_DATA}" ]]; then
    echo "Missing LIBERO_DATA: ${LIBERO_DATA}" >&2
    exit 1
fi
if [[ ! -e "${TEXT_EMBED_CACHE}" ]]; then
    echo "Missing TEXT_EMBED_CACHE: ${TEXT_EMBED_CACHE}" >&2
    echo "Set TEXT_EMBED_CACHE=... or generate/copy the old libero_umt5_text_embeds.pt cache." >&2
    exit 1
fi
if [[ -n "${FASTWAM_STATS}" && ! -f "${FASTWAM_STATS}" ]]; then
    echo "Missing FASTWAM_STATS: ${FASTWAM_STATS}" >&2
    exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Missing executable ACCELERATE_BIN: ${ACCELERATE_BIN}" >&2
    echo "Set ACCELERATE_BIN=... or activate the starVLA conda environment." >&2
    exit 1
fi

EFF_BS=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

echo "============================================================"
echo "WanFastWAM LIBERO v3-par3 repro training"
echo "============================================================"
echo "GPUs:           ${NUM_GPUS}"
echo "Per-device BS:  ${BATCH_SIZE}"
echo "Grad accum:     ${GRAD_ACCUM}"
echo "Effective BS:   ${EFF_BS}"
echo "Max steps:      ${MAX_STEPS}"
echo "Warmup steps:   ${WARMUP_STEPS}"
echo "Video backend:  ${VIDEO_BACKEND}"
echo "Base WM:        ${BASE_WM}"
echo "Data:           ${LIBERO_DATA}"
echo "Text cache:     ${TEXT_EMBED_CACHE}"
echo "FastWAM stats:  ${FASTWAM_STATS:-<disabled>}"
echo "Accelerate:     ${ACCELERATE_BIN}"
echo "Run ID:         ${run_id}"
echo "============================================================"

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

RESUME_ARGS=()
if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_ARGS=(--trainer.is_resume true --trainer.resume_step "${RESUME_STEP}")
    echo "Resuming from step ${RESUME_STEP}"
fi

"${ACCELERATE_BIN}" launch \
    --config_file "${DS_CONFIG_OVERRIDE:-starVLA/config/deepseeds/deepspeed_zero2_fastwam.yaml}" \
    --num_processes "${NUM_GPUS}" \
    --main_process_port "${MASTER_PORT}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name "${Framework_name}" \
    --framework.world_model.base_wm "${BASE_WM}" \
    --framework.qwenvl.base_vlm "${BASE_WM}" \
    --framework.world_model.text_embed_cache_path "${TEXT_EMBED_CACHE}" \
    --framework.world_model.zero_pad_text_embeds true \
    --framework.world_model.force_text_mask_ones true \
    --framework.action_dit.skip_pretrained_load false \
    --framework.fastwam.lambda_video 1.0 \
    --framework.fastwam.lambda_action 1.0 \
    --framework.fastwam.enable_video_loss true \
    --datasets.vla_data.data_root_dir "${LIBERO_DATA}" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --datasets.vla_data.fastwam_dataset_stats_path "${FASTWAM_STATS}" \
    --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
    --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
    --trainer.max_train_steps "${MAX_STEPS}" \
    --trainer.num_warmup_steps "${WARMUP_STEPS}" \
    --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --trainer.save_interval 10000 \
    --trainer.logging_frequency 100 \
    --trainer.eval_interval 100 \
    --trainer.enable_mixed_precision_training true \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    "${RESUME_ARGS[@]}" \
    "$@"
