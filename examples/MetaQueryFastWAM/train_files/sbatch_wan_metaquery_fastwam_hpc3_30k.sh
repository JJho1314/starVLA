#!/bin/bash
#SBATCH --job-name=metaq-wfw-30k
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/user/jhe724/workspace/starVLA/playground/Checkpoints/metaq_wfw_30k_%j.out
#SBATCH --error=/data/user/jhe724/workspace/starVLA/playground/Checkpoints/metaq_wfw_30k_%j.err

# ============================================================================
# WanMetaQueryFastWAM — 30k-step LIBERO training on HPC3.
#
# Compute budget:
#   8× H100 / A100 (acd_u),  effective batch = 128
#     per_device_batch_size : 2
#     gradient_accumulation : 8
#     num_processes         : 8
#     ⇒ 2 × 8 × 8 = 128
#
#   30,000 steps × 128 samples = 3.84 M sample-passes.
#   Wall-clock estimate: ~16–20 h on 8× H100.  SLURM cap: 36 h.
#
# Aligned with `run_libero_train_wanfastwam_v3par3_repro.sh`:
#   - uses deepspeed_zero2_fastwam.yaml (bucket-tuned for FastWAM)
#   - uses precomputed UMT5 text embed cache (skip 11 GB UMT5 / rank)
#   - same dataset_stats / data_mix conventions
#   - same WANDB self-hosted instance settings
# ============================================================================

set -euo pipefail

mkdir -p /data/user/jhe724/workspace/starVLA/playground/Checkpoints

echo "=== Job Info ==="
echo "JobID: ${SLURM_JOB_ID:-(no-slurm)}"
echo "Node:  ${SLURMD_NODENAME:-$(hostname)}"
echo "Date:  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | head

# ---------------------------------------------------------------------------
# Environment activation
# ---------------------------------------------------------------------------
source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/jhe724/.conda/envs/starVLA
export PATH=/data/user/jhe724/.conda/envs/starVLA/bin:$PATH
cd /data/user/jhe724/workspace/starVLA

# ---------------------------------------------------------------------------
# NCCL / torch warnings
# ---------------------------------------------------------------------------
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

# Silence the torchvision_av deprecation noise (matches v3par3 repro).
TORCHVISION_VIDEO_WARNING_FILTER="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"
if [[ -z "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${TORCHVISION_VIDEO_WARNING_FILTER}"
else
    export PYTHONWARNINGS="${PYTHONWARNINGS},${TORCHVISION_VIDEO_WARNING_FILTER}"
fi

# ---------------------------------------------------------------------------
# W&B (HPC3 self-hosted at 10.12.1.245:8080)
# ---------------------------------------------------------------------------
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.12.1.245:8080}"
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set (export it in your env, e.g. ~/.bashrc)}"
export WANDB_ENTITY="${WANDB_ENTITY:-jjho1314}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Libero}"

# ---------------------------------------------------------------------------
# Path defaults (HPC3 first, LFT-W02 fallback — same dual-detection as v3par3)
# ---------------------------------------------------------------------------
if [[ -d /data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers ]]; then
    default_base_wm=/data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers
else
    default_base_wm=/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers
fi

if [[ -d /data/user/jhe724/workspace/weights/Qwen3-VL-4B-Instruct ]]; then
    default_base_vlm=/data/user/jhe724/workspace/weights/Qwen3-VL-4B-Instruct
else
    default_base_vlm=/data/LFT-W02_data/junjie/weights/Qwen3-VL-4B-Instruct
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
elif [[ -d /data/LFT-W02_data/junjie/weights/fastwam_text_cache_safetensors_libero ]]; then
    default_text_cache=/data/LFT-W02_data/junjie/weights/fastwam_text_cache_safetensors_libero
else
    default_text_cache=""
fi

if [[ -f /data/user/jhe724/workspace/FastWAM/runs/libero_uncond_2cam224_1e-4/2026-05-09_16-19-41/dataset_stats.json ]]; then
    default_fastwam_stats=/data/user/jhe724/workspace/FastWAM/runs/libero_uncond_2cam224_1e-4/2026-05-09_16-19-41/dataset_stats.json
elif [[ -f /data/LFT-W02_data/junjie/VLA_WM/FastWAM/checkpoints/fastwam_release/fastwam/libero_uncond_2cam224_dataset_stats.json ]]; then
    default_fastwam_stats=/data/LFT-W02_data/junjie/VLA_WM/FastWAM/checkpoints/fastwam_release/fastwam/libero_uncond_2cam224_dataset_stats.json
else
    default_fastwam_stats=""
fi

# ---------------------------------------------------------------------------
# Knobs (all overridable via env)
# ---------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
BASE_WM="${BASE_WM:-${default_base_wm}}"
BASE_VLM="${BASE_VLM:-${default_base_vlm}}"
LIBERO_DATA="${LIBERO_DATA:-${default_libero_data}}"
TEXT_EMBED_CACHE="${TEXT_EMBED_CACHE:-${default_text_cache}}"
FASTWAM_STATS="${FASTWAM_STATS:-${default_fastwam_stats}}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/user/jhe724/.conda/envs/starVLA/bin/accelerate}"
DS_CONFIG="${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_fastwam.yaml}"

# Effective batch = 8 × 2 × 8 = 128 (matches the canonical sweep grid).
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-30000}"
# Warmup scaled down from WanFastWAM's 1085 at 80k → ~400 at 30k.
WARMUP_STEPS="${WARMUP_STEPS:-400}"
RESUME_STEP="${RESUME_STEP:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
VLM_ATTN="${VLM_ATTN:-flash_attention_2}"
# ── Validation knobs ── (0 to disable val)
VAL_FRACTION="${VAL_FRACTION:-0.02}"   # hold out 2 % for val
VAL_INTERVAL="${VAL_INTERVAL:-500}"    # run val every N optim steps
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-8}"

# ---------------------------------------------------------------------------
# Pre-flight sanity checks
# ---------------------------------------------------------------------------
err=0
[[ -d "${BASE_WM}"  ]]                || { echo "✗ Missing BASE_WM: ${BASE_WM}" >&2; err=1; }
[[ -d "${BASE_VLM}" ]]                || { echo "✗ Missing BASE_VLM: ${BASE_VLM}" >&2; err=1; }
[[ -d "${LIBERO_DATA}" ]]             || { echo "✗ Missing LIBERO_DATA: ${LIBERO_DATA}" >&2; err=1; }
[[ -e "${TEXT_EMBED_CACHE}" ]]        || { echo "✗ Missing TEXT_EMBED_CACHE: ${TEXT_EMBED_CACHE}" >&2; err=1; }
[[ -f "${FASTWAM_STATS}" ]]           || { echo "✗ Missing FASTWAM_STATS: ${FASTWAM_STATS}" >&2; err=1; }
[[ -x "${ACCELERATE_BIN}" ]]          || { echo "✗ Missing ACCELERATE_BIN: ${ACCELERATE_BIN}" >&2; err=1; }
[[ -f "${DS_CONFIG}" ]]               || { echo "✗ Missing DS_CONFIG: ${DS_CONFIG}" >&2; err=1; }
(( err == 0 )) || { echo "Pre-flight failed."; exit 1; }

# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------
Framework_name=WanMetaQueryFastWAM
config_yaml=./examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero.yaml
data_mix=libero_all_fastwam
run_root_dir=./playground/Checkpoints
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-nojob}_libero_${Framework_name}_30k_bs128}"
EFF_BS=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

echo "============================================================"
echo "WanMetaQueryFastWAM — LIBERO  (30k steps, eff_bs=${EFF_BS})"
echo "============================================================"
echo "GPUs:           ${NUM_GPUS}"
echo "Per-device BS:  ${BATCH_SIZE}"
echo "Grad accum:     ${GRAD_ACCUM}"
echo "Effective BS:   ${EFF_BS}"
echo "Max steps:      ${MAX_STEPS}"
echo "Warmup steps:   ${WARMUP_STEPS}"
echo "Base WM:        ${BASE_WM}"
echo "Base VLM:       ${BASE_VLM}"
echo "Data:           ${LIBERO_DATA}"
echo "Text cache:     ${TEXT_EMBED_CACHE}"
echo "FastWAM stats:  ${FASTWAM_STATS}"
echo "DS config:      ${DS_CONFIG}"
echo "VLM attn:       ${VLM_ATTN}"
echo "Run ID:         ${run_id}"
echo "============================================================"

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# ---------------------------------------------------------------------------
# Resume args
# ---------------------------------------------------------------------------
RESUME_ARGS=()
if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_ARGS=(--trainer.is_resume true --trainer.resume_step "${RESUME_STEP}")
    echo "Resuming from step ${RESUME_STEP}"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
"${ACCELERATE_BIN}" launch \
    --config_file "${DS_CONFIG}" \
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
    --framework.vlm.base_vlm "${BASE_VLM}" \
    --framework.vlm.attn_implementation "${VLM_ATTN}" \
    --datasets.vla_data.data_root_dir "${LIBERO_DATA}" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --datasets.vla_data.fastwam_dataset_stats_path "${FASTWAM_STATS}" \
    --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
    --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
    --trainer.max_train_steps "${MAX_STEPS}" \
    --trainer.num_warmup_steps "${WARMUP_STEPS}" \
    --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --trainer.freeze_modules 'backbone.vae,vlm' \
    --trainer.save_interval 5000 \
    --trainer.eval_interval 200 \
    --trainer.val_interval "${VAL_INTERVAL}" \
    --trainer.val_num_batches "${VAL_NUM_BATCHES}" \
    --datasets.vla_data.val_fraction "${VAL_FRACTION}" \
    --trainer.logging_frequency 50 \
    --trainer.enable_mixed_precision_training true \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    "${RESUME_ARGS[@]}" \
    "$@"

echo "=== Job Done $(date) ==="
