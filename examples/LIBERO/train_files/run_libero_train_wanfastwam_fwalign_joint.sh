#!/usr/bin/env bash
# WanFastWAM LIBERO training — FastWAM-aligned IO + JOINT MoT attention.
#
# Sibling of `run_libero_train_wanfastwam_fwalign.sh` (the 97.05 fwalign recipe).
# Two divergences vs that script:
#   - config_yaml = starvla_wanfastwam_libero_fwalign_joint.yaml
#     (carries `framework.fastwam.mot_attention_mode: joint` + num_video_frames)
#   - run_id default has `_joint` suffix so ckpt dir is distinct
#   - belt+suspenders CLI override: `--framework.fastwam.mot_attention_mode joint`
#     forced regardless of which yaml the caller passes via CONFIG_YAML
#
# Same env-var contract / accelerate launcher / paths as the fwalign sibling.
# The yaml swap rewires `build_mot_attention_mask` so that action attends to
# the FULL video latent during training. Resulting ckpts must be evaluated with
# `predict_action_joint` (via run_eval_fwalign_joint_faithful_local.sh) — the
# standard `predict_action` path (1-prefill + KV cache reuse) would feed the
# action expert a smaller K/V than it learned to consume.
#
# Typical HPC3 launch (after `srun` allocation):
#   tmux new -s train 'bash examples/LIBERO/train_files/run_libero_train_wanfastwam_fwalign_joint.sh'
#
# Useful overrides:
#   NUM_GPUS=8 bash examples/LIBERO/train_files/run_libero_train_wanfastwam_fwalign_joint.sh
#   MAX_STEPS=21700 BATCH_SIZE=8 GRAD_ACCUM=2 bash ...

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
config_yaml="${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_wanfastwam_libero_fwalign_joint.yaml}"
data_mix="${DATA_MIX:-libero_all_fastwam}"
run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_libero_${Framework_name}_fwalign_joint}"

# === HPC3 vs LFT-W02 default paths ===
if [[ -d /data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers ]]; then
    default_base_wm=/data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers
else
    default_base_wm=/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers
fi
if [[ -d /data/user/jhe724/workspace/FastWAM/data/libero_mujoco3.3.2 ]]; then
    default_libero_data=/data/user/jhe724/workspace/FastWAM/data/libero_mujoco3.3.2
else
    default_libero_data=/data/LFT-W02_data/junjie/data/libero_datasets/libero_lerobot
fi
if [[ -d /data/user/jhe724/workspace/FastWAM ]]; then
    default_fastwam_repo=/data/user/jhe724/workspace/FastWAM
else
    default_fastwam_repo=/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean
fi
if [[ -d /data/user/jhe724/workspace/FastWAM/checkpoints ]]; then
    default_fastwam_ckpts=/data/user/jhe724/workspace/FastWAM/checkpoints
else
    default_fastwam_ckpts=/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean/checkpoints
fi
if [[ -f /data/user/jhe724/workspace/FastWAM/runs/libero_uncond_2cam224_1e-4/2026-05-09_16-19-41/dataset_stats.json ]]; then
    default_fastwam_stats=/data/user/jhe724/workspace/FastWAM/runs/libero_uncond_2cam224_1e-4/2026-05-09_16-19-41/dataset_stats.json
else
    default_fastwam_stats=""
fi
# NOTE: when use_fastwam_aligned_io=true the loader writes text-embeds at
# encode-time via DiffSynth's WanTextEncoder. If you've pre-generated a
# cache that was made by the DiffSynth tokenizer (NOT the diffusers UMT5),
# point at it here to skip loading T5 at train time.
if [[ -d /data/user/jhe724/workspace/FastWAM/data/text_embeds_cache/libero ]]; then
    default_text_cache=/data/user/jhe724/workspace/FastWAM/data/text_embeds_cache/libero
else
    default_text_cache=/data/LFT-W02_data/junjie/weights/fastwam_text_cache_safetensors_libero
fi

NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/user/jhe724/.conda/envs/starVLA/bin/accelerate}"
BASE_WM="${BASE_WM:-${default_base_wm}}"
LIBERO_DATA="${LIBERO_DATA:-${default_libero_data}}"
TEXT_EMBED_CACHE="${TEXT_EMBED_CACHE:-${default_text_cache}}"
FASTWAM_STATS="${FASTWAM_STATS:-${default_fastwam_stats}}"

# Wan2_fastwam needs these at import time
export STARVLA_FASTWAM_REPO_PATH="${STARVLA_FASTWAM_REPO_PATH:-${default_fastwam_repo}}"
export STARVLA_FASTWAM_CHECKPOINTS_ROOT="${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-${default_fastwam_ckpts}}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_STEPS="${MAX_STEPS:-21700}"
WARMUP_STEPS="${WARMUP_STEPS:-1085}"
RESUME_STEP="${RESUME_STEP:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"

if [[ ! -d "${BASE_WM}" ]]; then
    echo "Missing BASE_WM: ${BASE_WM}" >&2 ; exit 1
fi
if [[ ! -d "${LIBERO_DATA}" ]]; then
    echo "Missing LIBERO_DATA: ${LIBERO_DATA}" >&2 ; exit 1
fi
if [[ ! -e "${TEXT_EMBED_CACHE}" ]]; then
    echo "Missing TEXT_EMBED_CACHE: ${TEXT_EMBED_CACHE}" >&2
    echo "Set TEXT_EMBED_CACHE=... or generate with starVLA/scripts/regen_text_cache.py" >&2
    exit 1
fi
if [[ -n "${FASTWAM_STATS}" && ! -f "${FASTWAM_STATS}" ]]; then
    echo "Missing FASTWAM_STATS: ${FASTWAM_STATS}" >&2 ; exit 1
fi
if [[ ! -d "${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" ]]; then
    echo "Missing FastWAM repo src/fastwam/ at: ${STARVLA_FASTWAM_REPO_PATH}/src/fastwam" >&2
    echo "Clone github.com/yuantianyuan01/FastWAM and set STARVLA_FASTWAM_REPO_PATH=..." >&2
    exit 1
fi
if [[ ! -d "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio" ]]; then
    echo "Missing DiffSynth-Studio dir at: ${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio" >&2
    echo "Download Wan2.2_VAE.safetensors + models_t5_umt5-xxl-enc-bf16.safetensors first." >&2
    exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Missing executable ACCELERATE_BIN: ${ACCELERATE_BIN}" >&2 ; exit 1
fi

EFF_BS=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

echo "============================================================"
echo "WanFastWAM LIBERO training — FASTWAM-ALIGNED IO (fwalign)"
echo "============================================================"
echo "GPUs:               ${NUM_GPUS}"
echo "Per-device BS:      ${BATCH_SIZE}"
echo "Grad accum:         ${GRAD_ACCUM}"
echo "Effective BS:       ${EFF_BS}"
echo "Max steps:          ${MAX_STEPS}"
echo "Warmup steps:       ${WARMUP_STEPS}"
echo "Video backend:      ${VIDEO_BACKEND}"
echo "Base WM (diffusers):${BASE_WM}"
echo "Data:               ${LIBERO_DATA}"
echo "Text cache:         ${TEXT_EMBED_CACHE}"
echo "FastWAM stats:      ${FASTWAM_STATS:-<disabled>}"
echo "FastWAM repo:       ${STARVLA_FASTWAM_REPO_PATH}"
echo "FastWAM ckpts:      ${STARVLA_FASTWAM_CHECKPOINTS_ROOT}"
echo "Accelerate:         ${ACCELERATE_BIN}"
echo "Run ID:             ${run_id}"
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
    --framework.world_model.use_fastwam_aligned_io true \
    --framework.world_model.fastwam_checkpoints_root "${STARVLA_FASTWAM_CHECKPOINTS_ROOT}" \
    --framework.world_model.redirect_common_files true \
    --framework.action_dit.skip_pretrained_load false \
    --framework.fastwam.lambda_video 1.0 \
    --framework.fastwam.lambda_action 1.0 \
    --framework.fastwam.enable_video_loss true \
    --framework.fastwam.mot_attention_mode joint \
    --framework.fastwam.num_video_frames 9 \
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
