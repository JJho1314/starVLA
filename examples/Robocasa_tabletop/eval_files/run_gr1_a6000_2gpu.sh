#!/bin/bash
# Robocasa GR1 WanFastWAM eval on local A6000 (2 GPUs, 49 GB each).
#
# Server-client like the LIBERO-plus pipeline: 2 model servers (1 per A6000)
# share inference across 24 concurrent sim clients (one per GR1 PnP env),
# round-robin to the 2 servers. Request batching keeps both GPUs busy while
# the 24 mujoco sims run on CPU. A6000's 49 GB easily holds the ~13 GB model
# + ~12 EGL render contexts per GPU.
#
# Two conda envs (separate because robosuite versions differ):
#   - starVLA env       -> model server (torch 2.6)
#   - robocasa_starVLA  -> sim client (robocasa GR1 fork + robosuite 1.5.2 + mujoco 3.2.6)
#
# Usage:
#   CKPT=/data/.../gr1_ola_ckpts/final_model/pytorch_model.pt bash run_gr1_a6000_2gpu.sh
set -uo pipefail  # NOT -e: a transient nonzero must not abort the launcher (would SIGHUP-kill all clients + trap-kill servers)

##### Paths #####
STARVLA_DIR=${STARVLA_DIR:-/data/LFT-W02_data/junjie/VLA_WM/starVLA}
STARVLA_PY=${STARVLA_PY:-/data/LFT-W02_data/.conda/envs/starVLA/bin/python}
ROBOCASA_PY=${ROBOCASA_PY:-/data/LFT-W02_data/.conda/envs/robocasa_starVLA/bin/python}
CKPT=${CKPT:-/data/LFT-W02_data/junjie/VLA_WM/gr1_ola_ckpts/final_model/pytorch_model.pt}
# dataset_statistics.json + config.yaml live in the RUN dir, which is the
# PARENT of final_model/ (ckpt is at <run_dir>/final_model/pytorch_model.pt).
RUN_DIR=${RUN_DIR:-$(dirname "$(dirname "$CKPT")")}
BASE_WM=${BASE_WM:-/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers}
DIFFSYNTH_BASE=${DIFFSYNTH_BASE:-/data/LFT-W02_data/junjie/weights}
OUTPUT_DIR=${OUTPUT_DIR:-/data/LFT-W02_data/junjie/VLA_WM/eval_runs/gr1_a6000_$(date +%Y%m%d_%H%M%S)}

##### Eval tuning #####
NUM_GPUS=${NUM_GPUS:-2}
N_EPISODES=${N_EPISODES:-50}
N_ENVS=${N_ENVS:-1}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-720}
N_ACTION_STEPS=${N_ACTION_STEPS:-12}
BATCH_SIZE=${BATCH_SIZE:-12}        # server-side max batch (≈ clients/server)
BATCH_WAIT_MS=${BATCH_WAIT_MS:-20}
BASE_PORT=${BASE_PORT:-6398}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR/server_logs" "$OUTPUT_DIR/client_logs" "$OUTPUT_DIR/videos"

echo "==== Robocasa GR1 eval on A6000 (${NUM_GPUS} GPU) ===="
echo "  CKPT      = $CKPT"
echo "  BASE_WM   = $BASE_WM"
echo "  OUTPUT    = $OUTPUT_DIR"
echo "  N_EPISODES=$N_EPISODES  MAX_STEPS=$MAX_EPISODE_STEPS  N_ACTION_STEPS=$N_ACTION_STEPS"
echo "  BATCH_SIZE=$BATCH_SIZE  BATCH_WAIT_MS=$BATCH_WAIT_MS"
echo

cd "$STARVLA_DIR"
HOST=127.0.0.1

ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

##### 1) Launch model servers (starVLA env) #####
declare -a SERVER_PIDS=()
for g in $(seq 0 $((NUM_GPUS - 1))); do
  port=$((BASE_PORT + g))
  echo "[$(date +%H:%M:%S)] server gpu=$g port=$port"
  CUDA_VISIBLE_DEVICES=$g "$STARVLA_PY" \
      deployment/model_server/server_wanfastwam_gr1_a6000.py \
      --ckpt_path "$CKPT" --run_dir "$RUN_DIR" \
      --base_wm "$BASE_WM" --diffsynth_base "$DIFFSYNTH_BASE" \
      --port $port --use_bf16 --idle_timeout -1 \
      --batch_size $BATCH_SIZE --batch_wait_ms $BATCH_WAIT_MS \
      >"$OUTPUT_DIR/server_logs/server_gpu${g}_p${port}.log" 2>&1 &
  SERVER_PIDS+=($!)
done
trap 'echo cleanup; kill -9 "${SERVER_PIDS[@]}" 2>/dev/null' EXIT

# Wait for both servers to bind their ports.
for g in $(seq 0 $((NUM_GPUS - 1))); do
  port=$((BASE_PORT + g))
  for _ in $(seq 1 1800); do
    (echo > /dev/tcp/$HOST/$port) 2>/dev/null && { echo "server :$port READY"; break; }
    sleep 2
  done
done
echo "[$(date +%H:%M:%S)] all servers READY; launching ${#ENV_NAMES[@]} sim clients"

##### 2) Launch all sim clients (robocasa env), round-robin to servers #####
declare -a CLIENT_PIDS=()
idx=0
for env in "${ENV_NAMES[@]}"; do
  g=$((idx % NUM_GPUS))
  port=$((BASE_PORT + g))
  short=$(echo "$env" | sed 's#.*/##; s/_GR1.*//')
  vout="$OUTPUT_DIR/videos/$short"
  log="$OUTPUT_DIR/client_logs/${short}.log"
  mkdir -p "$vout"
  # Render the mujoco sim on CPU (osmesa), NOT GPU EGL. With 24 clients,
  # simultaneous EGL context init on the 2 server GPUs corrupted/OOM'd the
  # server CUDA contexts (servers got SIGKILLed right when clients launched).
  # osmesa keeps clients 100% off-GPU so the servers own the GPUs exclusively;
  # the box has 1 TB RAM + many cores for 24 CPU sims. CUDA_VISIBLE_DEVICES=""
  # ensures the client never touches a GPU (it only does websocket + mujoco).
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    "$ROBOCASA_PY" examples/Robocasa_tabletop/eval_files/simulation_env.py \
      --args.host "$HOST" --args.port $port \
      --args.env_name "$env" \
      --args.n_episodes $N_EPISODES \
      --args.n_envs $N_ENVS \
      --args.max_episode_steps $MAX_EPISODE_STEPS \
      --args.n_action_steps $N_ACTION_STEPS \
      --args.video_out_path "$vout" \
      --args.pretrained_path "$CKPT" \
      >"$log" 2>&1 &
  CLIENT_PIDS+=($!)
  idx=$((idx + 1))
  sleep 1
done

echo "[$(date +%H:%M:%S)] launched ${#CLIENT_PIDS[@]} sim clients"
for pid in "${CLIENT_PIDS[@]}"; do wait "$pid" || true; done

echo "[$(date +%H:%M:%S)] all sim clients done. Aggregating ..."
"$STARVLA_PY" - <<PY
import glob, json, os, re
root = "$OUTPUT_DIR"
# Each client logs final success rate; parse from client logs.
results = {}
for f in glob.glob(os.path.join(root, "client_logs", "*.log")):
    name = os.path.basename(f)[:-4]
    txt = open(f, errors="ignore").read()
    m = re.findall(r"[Ss]uccess[_ ]?rate[:=]\s*([0-9.]+)", txt)
    sr = float(m[-1]) if m else None
    # also try "X/Y" style
    results[name] = sr
ok = [v for v in results.values() if v is not None]
print("=== Robocasa GR1 per-env success rate ===")
for k in sorted(results):
    print(f"  {k}: {results[k]}")
if ok:
    print(f"MEAN over {len(ok)} envs: {sum(ok)/len(ok):.4f}")
json.dump(results, open(os.path.join(root, "aggregate.json"), "w"), indent=2)
PY
echo "[$(date +%H:%M:%S)] aggregate.json written to $OUTPUT_DIR"
