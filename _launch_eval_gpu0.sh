#!/usr/bin/env bash
# One-shot launcher for WanMetaQueryFastWAM eval on LFT-W02 GPU 0 only.
# GPU 1 is left free for other tasks.
set -uo pipefail

cd /data/LFT-W02_data/junjie/VLA_WM/starVLA

RUN_DIR=/data/LFT-W02_data/junjie/VLA_WM/starVLA/playground/Checkpoints/20260520_220415_308038_libero_WanMetaQueryFastWAM_30k_bs128

# Kill leftovers + clear stale state.
pkill -9 -f "eval_all_parallel_fwalign_joint_faithful" 2>/dev/null
pkill -9 -f "server_wanfastwam_starvla_joint_faithful" 2>/dev/null
pkill -9 -f "eval_libero\.py" 2>/dev/null
sleep 2

: > "$RUN_DIR/eval_launch.log"
rm -rf "$RUN_DIR/eval_logs_joint_faithful"
mkdir -p "$RUN_DIR/eval_logs_joint_faithful"

export FWALIGN_RUN_DIR=$RUN_DIR
export CKPT=$RUN_DIR/final_model/pytorch_model.pt
export STARVLA_FASTWAM_REPO_PATH=/data/LFT-W02_data/junjie/VLA_WM/FastWAM
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=$STARVLA_FASTWAM_REPO_PATH/checkpoints
export DIFFSYNTH_MODEL_BASE_PATH=$STARVLA_FASTWAM_CHECKPOINTS_ROOT
export DIFFSYNTH_SKIP_DOWNLOAD=true
export N_TRIALS=10
export SUITES="libero_spatial libero_object libero_goal libero_10"
export BATCH_SIZE=5
export BATCH_WAIT_MS=20
export N_WORKERS_PER_SUITE=2
export TASKS_PER_WORKER=5

echo "[$(date)] starting eval on GPU 0 (sam3 already takes 13.8GB; eval needs ~25GB; tight but fits)" > "$RUN_DIR/eval_launch.log"
exec bash examples/LIBERO/eval_files/eval_all_parallel_fwalign_joint_faithful_gpu0only.sh >> "$RUN_DIR/eval_launch.log" 2>&1
