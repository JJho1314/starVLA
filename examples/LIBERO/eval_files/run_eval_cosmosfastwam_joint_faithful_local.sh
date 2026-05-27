#!/usr/bin/env bash
# LIBERO eval — Cosmos FastWAM joint-FAITHFUL inference.
#
# Sibling of run_eval_fwalign_joint_faithful_local.sh. Same V8 pipeline +
# joint-faithful inference (predict_action_joint, per-step full MoT), but
# the run dir contains a Cosmos-trained ckpt (CosmoPredict_fastwam backbone)
# instead of a Wan-trained ckpt.
#
# Since the dispatcher routes on the saved config.yaml's
# ``framework.world_model.base_wm`` field, the existing server file
# (server_wanfastwam_starvla_joint_faithful.py) and eval_libero.py client
# work without any modification — they delegate everything to
# baseframework.from_pretrained(ckpt_path) which routes correctly.
#
# Usage:
#   FWALIGN_RUN_DIR=/path/to/cosmos_run_dir \\
#     bash examples/LIBERO/eval_files/run_eval_cosmosfastwam_joint_faithful_local.sh

set -u

# Point at the Cosmos-trained run directory.
FWALIGN_RUN_DIR=${FWALIGN_RUN_DIR:-/data/LFT-W02_data/junjie/VLA_WM/cosmosjoint_olabots_ckpts}
export CKPT=${CKPT:-${FWALIGN_RUN_DIR}/final_model/pytorch_model.pt}

# Cosmos backbone uses diffusers components directly — no DiffSynth env vars
# needed. But the existing eval_all_parallel_fwalign_joint_faithful.sh still
# tries to validate ${STARVLA_FASTWAM_CHECKPOINTS_ROOT}/DiffSynth-Studio path
# (inherited from Wan-side eval). Stub it to a directory that exists so the
# check passes; the Cosmos backbone never reads it.
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-${STARVLA_FASTWAM_REPO_PATH}/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-${STARVLA_FASTWAM_CHECKPOINTS_ROOT}}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}

export N_TRIALS=${N_TRIALS:-50}
export SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
export BATCH_SIZE=${BATCH_SIZE:-5}
export BATCH_WAIT_MS=${BATCH_WAIT_MS:-20}

if [[ -f "${FWALIGN_RUN_DIR}/config.yaml" ]]; then
    if ! grep -q "cosmos-predict" "${FWALIGN_RUN_DIR}/config.yaml" 2>/dev/null; then
        echo "[warn] ${FWALIGN_RUN_DIR}/config.yaml does NOT seem to be Cosmos-trained" >&2
        echo "       (no 'cosmos-predict' in base_wm). This launcher expects Cosmos." >&2
    fi
fi

for p in "${CKPT}" "${FWALIGN_RUN_DIR}/config.yaml" "${FWALIGN_RUN_DIR}/dataset_statistics.json"; do
    [[ -e "$p" ]] || { echo "[fatal] missing: $p" >&2; exit 1; }
done
echo "[ok] CKPT=${CKPT}"
echo "[ok] FWALIGN_RUN_DIR=${FWALIGN_RUN_DIR}"
echo "[ok] backbone: Cosmos-Predict2 (routed via base_wm containing 'cosmos-predict')"
echo "[ok] inference path: joint_faithful (predict_action_joint, per-step full MoT)"
echo "[ok] SUITES=${SUITES}  N_TRIALS=${N_TRIALS}  BATCH_SIZE=${BATCH_SIZE}"

exec bash "$(dirname "$0")/eval_all_parallel_fwalign_joint_faithful.sh" "$@"
