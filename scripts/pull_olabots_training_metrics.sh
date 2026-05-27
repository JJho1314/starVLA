#!/usr/bin/env bash
# Pull WanFastWAM training launch log (and wandb dir) from olabots, parse it
# into CSV + PNG locally for IDE viewing.
#
# Defaults target the fwjoint run launched on 2026-05-19 15:07 UTC; override
# via env vars to point at a different run.
#
# Usage:
#   bash scripts/pull_olabots_training_metrics.sh
#   RUN_ID=20260519_150709_libero_WanFastWAM_fwalign_joint bash scripts/pull_olabots_training_metrics.sh
#   OUT_DIR=~/myrun bash scripts/pull_olabots_training_metrics.sh
#   WATCH=1 bash scripts/pull_olabots_training_metrics.sh    # rerun every 5 min until killed
#
# Outputs (under $OUT_DIR, default ~/joint_metrics):
#   launch.log    — rsync'd raw trainer log
#   metrics.csv   — tabular (step, epoch, lr, action_loss, video_loss, mse_score, data_time, model_time)
#   metrics.png   — 4-panel matplotlib plot
#   wandb/        — rsync'd offline wandb dir (for `wandb sync` later if you set WANDB_API_KEY)

set -euo pipefail

# === Defaults — override via env ===
RUN_ID="${RUN_ID:-20260519_150709_libero_WanFastWAM_fwalign_joint}"
LOG_BASENAME="${LOG_BASENAME:-joint_launch_20260519_150709.log}"
OLABOTS_BASE="${OLABOTS_BASE:-/data/users/junjie/starVLA/playground/Checkpoints}"
OUT_DIR="${OUT_DIR:-$HOME/joint_metrics}"
SSH_HOST="${SSH_HOST:-olabots}"
PYTHON_BIN="${PYTHON_BIN:-/data/LFT-W02_data/.conda/envs/starVLA/bin/python}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"   # seconds; only used when WATCH=1

mkdir -p "$OUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

pull_once() {
    local ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] [1/3] rsync launch log + wandb from ${SSH_HOST}…"
    rsync -a --info=stats0,progress0 \
        "${SSH_HOST}:${OLABOTS_BASE}/${LOG_BASENAME}" \
        "${OUT_DIR}/launch.log"
    rsync -a --info=stats0,progress0 --delete \
        "${SSH_HOST}:${OLABOTS_BASE}/${RUN_ID}/wandb/" \
        "${OUT_DIR}/wandb/" \
        || echo "[warn] wandb dir rsync failed (run dir maybe not yet created); skipping"

    echo "[$ts] [2/3] parse metrics …"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/parse_olabots_training_metrics.py" \
        --log "${OUT_DIR}/launch.log" \
        --csv "${OUT_DIR}/metrics.csv" \
        --png "${OUT_DIR}/metrics.png" \
        --title "${RUN_ID}"

    echo "[$ts] [3/3] done."
    echo "         CSV : ${OUT_DIR}/metrics.csv"
    echo "         PNG : ${OUT_DIR}/metrics.png"
    echo "         LOG : ${OUT_DIR}/launch.log"
}

if [[ "${WATCH:-0}" == "1" ]]; then
    echo "[watch] refreshing every ${WATCH_INTERVAL}s (Ctrl+C to stop)"
    while true; do
        pull_once || echo "[warn] iteration failed; will retry"
        sleep "${WATCH_INTERVAL}"
    done
else
    pull_once
fi
