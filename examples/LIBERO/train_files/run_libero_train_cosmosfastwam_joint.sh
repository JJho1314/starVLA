#!/usr/bin/env bash
# CosmosFastWAM LIBERO training — Cosmos-Predict2 backbone + FastWAM-style MoT.
#
# Sibling of `run_libero_train_wanfastwam_fwalign_joint.sh`. Same accelerate
# launcher, same framework (WanFastWAM), but yaml points at
# `starvla_cosmosfastwam_libero_joint.yaml` which routes the world-model
# dispatcher to `CosmoPredict_fastwam.WanVideoBackboneFastWAMCosmos`.
#
# *** SCAFFOLD STATUS ***
# Backbone load + text/VAE encoding work. Joint MoT forward will fail until
# CosmoPredict_fastwam exposes pre_dit/post_dit/per-block-forward in the
# FastWAM signature (see TODO in CosmoPredict_fastwam.py header).
#
# Smoke pattern recommended before full training:
#   bash this_script.sh --datasets.vla_data.per_device_batch_size 1 \
#                       --trainer.max_train_steps 5 \
#                       --trainer.save_interval 100000
# Catches backbone load issues without spending GPU-hours.
#
# Usage (after smoke):
#   tmux new -s cosmosfw 'bash examples/LIBERO/train_files/run_libero_train_cosmosfastwam_joint.sh'

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
config_yaml="${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_cosmosfastwam_libero_joint.yaml}"
data_mix="${DATA_MIX:-libero_all_fastwam}"
run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_libero_${Framework_name}_cosmos_joint}"

# --- Default paths -------------------------------------------------------
# Cosmos-Predict2 lives on HuggingFace under "nvidia/Cosmos-Predict2-2B-Video2World"
# but you'll usually want a local clone for reliability. Two layers of override:
#   - BASE_WM env var → wins over yaml
#   - yaml world_model.base_wm
if [[ -d /data/shared/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World ]]; then
    default_base_wm=/data/shared/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World
elif [[ -d /data/LFT-W02_data/junjie/weights/Cosmos-Predict2-2B-Video2World ]]; then
    default_base_wm=/data/LFT-W02_data/junjie/weights/Cosmos-Predict2-2B-Video2World
else
    default_base_wm="nvidia/Cosmos-Predict2-2B-Video2World"
fi

if [[ -d /data/user/jhe724/workspace/FastWAM/data/libero_mujoco3.3.2 ]]; then
    default_libero_data=/data/user/jhe724/workspace/FastWAM/data/libero_mujoco3.3.2
else
    default_libero_data=/data/LFT-W02_data/junjie/data/libero_datasets/libero_lerobot
fi

# Stats: Cosmos has different VAE-space than Wan, so we DON'T reuse the
# FastWAM stats. Until a Cosmos stats file is generated, leave empty (means
# stats will be normalized on-the-fly by the dataloader's running estimate).
default_fastwam_stats=""

NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/users/wchen/envs/starvla/py310/bin/accelerate}"
BASE_WM="${BASE_WM:-${default_base_wm}}"
LIBERO_DATA="${LIBERO_DATA:-${default_libero_data}}"
FASTWAM_STATS="${FASTWAM_STATS:-${default_fastwam_stats}}"

# Optional pre-computed Cosmos T5 text cache (set to skip live T5 at train time)
TEXT_EMBED_CACHE="${TEXT_EMBED_CACHE:-}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_STEPS="${MAX_STEPS:-21700}"
WARMUP_STEPS="${WARMUP_STEPS:-1085}"
RESUME_STEP="${RESUME_STEP:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"

if [[ "${BASE_WM}" != "nvidia/Cosmos-Predict2-2B-Video2World" && ! -d "${BASE_WM}" ]]; then
    echo "[warn] BASE_WM=${BASE_WM} is neither a local dir nor the HF default;" >&2
    echo "       diffusers will attempt to download. Ctrl-C if that's wrong." >&2
fi
if [[ ! -d "${LIBERO_DATA}" ]]; then
    echo "Missing LIBERO_DATA: ${LIBERO_DATA}" >&2 ; exit 1
fi
if [[ -n "${FASTWAM_STATS}" && ! -f "${FASTWAM_STATS}" ]]; then
    echo "Missing FASTWAM_STATS: ${FASTWAM_STATS}" >&2 ; exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Missing executable ACCELERATE_BIN: ${ACCELERATE_BIN}" >&2 ; exit 1
fi

EFF_BS=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

echo "============================================================"
echo "CosmosFastWAM LIBERO training — Cosmos-Predict2 backbone"
echo "============================================================"
echo "GPUs:               ${NUM_GPUS}"
echo "Per-device BS:      ${BATCH_SIZE}"
echo "Grad accum:         ${GRAD_ACCUM}"
echo "Effective BS:       ${EFF_BS}"
echo "Max steps:          ${MAX_STEPS}"
echo "Warmup steps:       ${WARMUP_STEPS}"
echo "Video backend:      ${VIDEO_BACKEND}"
echo "Base WM (Cosmos):   ${BASE_WM}"
echo "Data:               ${LIBERO_DATA}"
echo "Text cache:         ${TEXT_EMBED_CACHE:-<none — T5 runs live>}"
echo "FastWAM stats:      ${FASTWAM_STATS:-<none — dataloader normalizes>}"
echo "Accelerate:         ${ACCELERATE_BIN}"
echo "Run ID:             ${run_id}"
echo "Status:             *** SCAFFOLD *** — joint MoT not yet wired"
echo "============================================================"

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

RESUME_ARGS=()
if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_ARGS=(--trainer.is_resume true --trainer.resume_step "${RESUME_STEP}")
    echo "Resuming from step ${RESUME_STEP}"
fi

TEXT_CACHE_ARGS=()
if [[ -n "${TEXT_EMBED_CACHE}" ]]; then
    TEXT_CACHE_ARGS=(--framework.world_model.text_embed_cache_path "${TEXT_EMBED_CACHE}")
fi

STATS_ARGS=()
if [[ -n "${FASTWAM_STATS}" ]]; then
    STATS_ARGS=(--datasets.vla_data.fastwam_dataset_stats_path "${FASTWAM_STATS}")
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
    --framework.world_model.zero_pad_text_embeds true \
    --framework.world_model.force_text_mask_ones true \
    --framework.world_model.use_fastwam_aligned_io true \
    --framework.action_dit.skip_pretrained_load true \
    --framework.fastwam.lambda_video 1.0 \
    --framework.fastwam.lambda_action 1.0 \
    --framework.fastwam.enable_video_loss true \
    --framework.fastwam.mot_attention_mode joint \
    --framework.fastwam.num_video_frames 9 \
    --datasets.vla_data.data_root_dir "${LIBERO_DATA}" \
    --datasets.vla_data.data_mix "${data_mix}" \
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
    "${TEXT_CACHE_ARGS[@]}" \
    "${STATS_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "$@"
