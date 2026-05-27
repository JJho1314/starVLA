#!/usr/bin/env bash
# Official LIBERO eval: N_TRIALS=50, 4 suites × 10 tasks × 50 trials = 2000 total.
# Uses both GPU 0 and GPU 1 in parallel (dual-server, matches canonical eval_all_parallel_*.sh).
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
# Also clear partial results so aggregator counts cleanly
rm -rf "$RUN_DIR/results" 2>/dev/null

export FWALIGN_RUN_DIR=$RUN_DIR
export CKPT=$RUN_DIR/final_model/pytorch_model.pt
export STARVLA_FASTWAM_REPO_PATH=/data/LFT-W02_data/junjie/VLA_WM/FastWAM
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=$STARVLA_FASTWAM_REPO_PATH/checkpoints
export DIFFSYNTH_MODEL_BASE_PATH=$STARVLA_FASTWAM_CHECKPOINTS_ROOT
export DIFFSYNTH_SKIP_DOWNLOAD=true
# Official LIBERO benchmark settings — matches canonical defaults
export N_TRIALS=50                                                   # ← official: 50 trials per task
export SUITES="libero_spatial libero_object libero_goal libero_10"
export BATCH_SIZE=5
export BATCH_WAIT_MS=20
# N_WORKERS_PER_SUITE * TASKS_PER_WORKER = 10 tasks per suite
export N_WORKERS_PER_SUITE=5
export TASKS_PER_WORKER=2

echo "[$(date)] starting OFFICIAL eval: N_TRIALS=50, dual GPU, 5 workers/suite × 4 suites = 20 LIBERO sims" > "$RUN_DIR/eval_launch.log"
# Use the unmodified canonical 2-GPU orchestrator (GPUS=(0 1), PORTS=(6694 6695))
exec bash examples/LIBERO/eval_files/eval_all_parallel_fwalign_joint_faithful.sh >> "$RUN_DIR/eval_launch.log" 2>&1
