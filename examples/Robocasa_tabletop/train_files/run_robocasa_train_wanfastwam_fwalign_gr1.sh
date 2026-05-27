#!/usr/bin/env bash
# WanFastWAM × Robocasa Fourier-GR1 (fwalign / non-joint mask) training.
#
# Sibling of examples/LIBERO/train_files/run_libero_train_wanfastwam_fwalign.sh.
# Differences:
#   - yaml = starvla_wanfastwam_robocasa_gr1_fwalign.yaml
#   - data_mix = fourier_gr1_unified_1000_fastwam (registered in
#     examples/Robocasa_tabletop/train_files/data_registry/data_config.py)
#   - action_dim=29 / state_dim=58 / single-cam 224x224 (yaml-side)
#   - ActionDiT pretrained warm-start NOT compatible (action_dim 7→29 mismatch);
#     yaml sets skip_pretrained_load=true. Forced again on CLI for clarity.
#
# Typical launch (after `srun` allocation):
#   tmux new -s gr1 'bash examples/Robocasa_tabletop/train_files/run_robocasa_train_wanfastwam_fwalign_gr1.sh'
#
# Useful overrides:
#   NUM_GPUS=8 bash this_script.sh
#   BATCH_SIZE=4 GRAD_ACCUM=4 MAX_STEPS=80000 bash this_script.sh
#   GR1_DATA=/path/to/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim bash this_script.sh

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
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Robocasa}"

Framework_name=WanFastWAM
config_yaml="${CONFIG_YAML:-./examples/Robocasa_tabletop/train_files/starvla_wanfastwam_robocasa_gr1_fwalign.yaml}"
data_mix="${DATA_MIX:-fourier_gr1_unified_1000_fastwam}"
run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_robocasa_${Framework_name}_gr1_fwalign}"

# === HPC3 vs LFT-W02 vs Olabots default paths ===
# BASE_WM: Wan2.2-TI2V-5B diffusers checkout (for tokenizer / model_index.json).
if [[ -d /data/shared/checkpoints/Wan-AI/Wan2.2-TI2V-5B-Diffusers ]]; then
    default_base_wm=/data/shared/checkpoints/Wan-AI/Wan2.2-TI2V-5B-Diffusers
elif [[ -d /data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers ]]; then
    default_base_wm=/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers
elif [[ -d /data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers ]]; then
    default_base_wm=/data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers
else
    default_base_wm=""
fi

# GR1 lerobot dataset (PhysicalAI-Robotics-GR00T-X-Embodiment-Sim).
if [[ -d /data/shared/datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim ]]; then
    default_gr1_data=/data/shared/datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
elif [[ -d /data/LFT-W02_data/junjie/data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim ]]; then
    default_gr1_data=/data/LFT-W02_data/junjie/data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
elif [[ -d /data/user/jhe724/workspace/data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim ]]; then
    default_gr1_data=/data/user/jhe724/workspace/data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
else
    default_gr1_data=""
fi

# FastWAM source repo + DiffSynth-Studio checkpoints (for use_fastwam_aligned_io).
if [[ -d /data/users/junjie/FastWAM/src/fastwam ]]; then
    default_fastwam_repo=/data/users/junjie/FastWAM
elif [[ -d /data/LFT-W02_data/junjie/VLA_WM/FastWAM/src/fastwam ]]; then
    default_fastwam_repo=/data/LFT-W02_data/junjie/VLA_WM/FastWAM
elif [[ -d /data/user/jhe724/workspace/FastWAM/src/fastwam ]]; then
    default_fastwam_repo=/data/user/jhe724/workspace/FastWAM
else
    default_fastwam_repo=""
fi
if [[ -d /data/shared/checkpoints/DiffSynth-Studio ]]; then
    default_fastwam_ckpts=/data/shared/checkpoints
elif [[ -d /data/LFT-W02_data/junjie/VLA_WM/FastWAM/checkpoints/DiffSynth-Studio ]]; then
    default_fastwam_ckpts=/data/LFT-W02_data/junjie/VLA_WM/FastWAM/checkpoints
elif [[ -d /data/user/jhe724/workspace/FastWAM/checkpoints/DiffSynth-Studio ]]; then
    default_fastwam_ckpts=/data/user/jhe724/workspace/FastWAM/checkpoints
else
    default_fastwam_ckpts=""
fi

NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/users/wchen/envs/starvla/py310/bin/accelerate}"
BASE_WM="${BASE_WM:-${default_base_wm}}"
GR1_DATA="${GR1_DATA:-${default_gr1_data}}"

# Wan2_fastwam needs these at import time
export STARVLA_FASTWAM_REPO_PATH="${STARVLA_FASTWAM_REPO_PATH:-${default_fastwam_repo}}"
export STARVLA_FASTWAM_CHECKPOINTS_ROOT="${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-${default_fastwam_ckpts}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

# Optional pre-computed Wan-T5 (UMT5-XXL) text cache for GR1 tasks.
TEXT_EMBED_CACHE="${TEXT_EMBED_CACHE:-}"

# 8 GPU × bs=4 × accum=4 = 128 effective (matches LIBERO fwalign 97.05 recipe).
# GR1 9-frame video at 224×224 is ~similar VRAM to LIBERO 9-frame at 224×448
# (same total tokens). Adjust per-GPU BS down on smaller cards.
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-80000}"
WARMUP_STEPS="${WARMUP_STEPS:-4000}"
RESUME_STEP="${RESUME_STEP:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"

# === Sanity checks ===
if [[ -z "${BASE_WM}" || ! -d "${BASE_WM}" ]]; then
    echo "[fatal] BASE_WM not found or empty: ${BASE_WM}" >&2
    echo "       set BASE_WM=/path/to/Wan2.2-TI2V-5B-Diffusers" >&2
    exit 1
fi
if [[ -z "${GR1_DATA}" || ! -d "${GR1_DATA}" ]]; then
    echo "[fatal] GR1_DATA not found: ${GR1_DATA}" >&2
    echo "       set GR1_DATA=/path/to/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim" >&2
    exit 1
fi
if [[ -z "${STARVLA_FASTWAM_REPO_PATH}" || ! -d "${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" ]]; then
    echo "[fatal] FastWAM repo not found at ${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" >&2
    exit 1
fi
if [[ ! -d "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio" ]]; then
    echo "[fatal] DiffSynth-Studio not found at ${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio" >&2
    exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "[fatal] ACCELERATE_BIN not executable: ${ACCELERATE_BIN}" >&2
    exit 1
fi

EFF_BS=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

echo "============================================================"
echo "WanFastWAM × Robocasa Fourier-GR1 — fwalign (non-joint) training"
echo "============================================================"
echo "GPUs:               ${NUM_GPUS}"
echo "Per-device BS:      ${BATCH_SIZE}"
echo "Grad accum:         ${GRAD_ACCUM}"
echo "Effective BS:       ${EFF_BS}"
echo "Max steps:          ${MAX_STEPS}"
echo "Warmup steps:       ${WARMUP_STEPS}"
echo "Video backend:      ${VIDEO_BACKEND}"
echo "Base WM:            ${BASE_WM}"
echo "GR1 data:           ${GR1_DATA}"
echo "Data mix:           ${data_mix}"
echo "Text cache:         ${TEXT_EMBED_CACHE:-<none — T5 runs live>}"
echo "FastWAM repo:       ${STARVLA_FASTWAM_REPO_PATH}"
echo "FastWAM ckpts:      ${STARVLA_FASTWAM_CHECKPOINTS_ROOT}"
echo "Accelerate:         ${ACCELERATE_BIN}"
echo "Run ID:             ${run_id}"
echo "Note:               ActionDiT pretrained skipped (action_dim 7→29 mismatch)"
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
    --framework.world_model.fastwam_checkpoints_root "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}" \
    --framework.world_model.redirect_common_files true \
    --framework.action_dit.skip_pretrained_load true \
    --framework.fastwam.lambda_video 1.0 \
    --framework.fastwam.lambda_action 1.0 \
    --framework.fastwam.enable_video_loss true \
    --datasets.vla_data.data_root_dir "${GR1_DATA}" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
    --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
    --trainer.max_train_steps "${MAX_STEPS}" \
    --trainer.num_warmup_steps "${WARMUP_STEPS}" \
    --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --trainer.save_interval 10000 \
    --trainer.logging_frequency 100 \
    --trainer.eval_interval 100 \
    --datasets.vla_data.val_fraction 0.05 \
    --trainer.val_interval 500 \
    --trainer.enable_mixed_precision_training true \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    "${TEXT_CACHE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "$@"
