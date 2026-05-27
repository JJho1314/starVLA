#!/bin/bash
# LIBERO-plus zeroshot eval, fwjoint final_model (V8 SR 98.30 on standard LIBERO).
# Designed for lg1 (8x RTX 4090, 24GB each).
#
# This ckpt was trained with `framework.fastwam.mot_attention_mode: joint`,
# so it requires `predict_action_joint` (full joint MoT forward) at inference
# — running the standard fast prefill+cache path would degrade SR.
# The worker's --joint_faithful flag selects that path.
#
# Layout (8 GPUs, 4 suites — each suite split across 2 GPUs):
#   GPU 0,1 -> libero_spatial   GPU 2,3 -> libero_object
#   GPU 4,5 -> libero_goal      GPU 6,7 -> libero_10
set -euo pipefail

##### Edit these paths #####
STARVLA_DIR=${STARVLA_DIR:-/data3/junjie/starVLA}
LIBERO_PLUS_HOME=${LIBERO_PLUS_HOME:-/data3/junjie/LIBERO-plus}
CONDA_PY=${CONDA_PY:-/data3/junjie/envs/starvla_eval/bin/python}
CKPT=${CKPT:-/data3/junjie/fwjoint_olabots_ckpts/final_model/pytorch_model.pt}
OUTPUT_DIR=${OUTPUT_DIR:-/data3/junjie/eval_runs/libero_plus_fwjoint_$(date +%Y%m%d_%H%M%S)}
# Override training-time num_inference_steps. fwjoint uses predict_action_joint
# (full MoT every step), so this scales linearly — 4 vs 10 = 2.5x speedup.
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-4}
# FastWAM repo + Wan ckpts (FastWAM-official safetensors path for V8 parity).
export STARVLA_FASTWAM_REPO_PATH=${STARVLA_FASTWAM_REPO_PATH:-/data3/junjie/FastWAM_official_clean}
export STARVLA_FASTWAM_CHECKPOINTS_ROOT=${STARVLA_FASTWAM_CHECKPOINTS_ROOT:-/data3/junjie/FastWAM_official_clean/checkpoints}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/data3/junjie}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}
# Pre-computed UMT5 text cache for all 10002 LIBERO-plus task descriptions.
export LIBERO_PLUS_TEXT_CACHE=${LIBERO_PLUS_TEXT_CACHE:-/data3/junjie/libero_plus_text_cache_lg1}
# Required to fit on 24 GB: dedupes fragmented free segments.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
##### End edit #####

export LIBERO_HOME="$LIBERO_PLUS_HOME"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_PLUS_HOME}:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR"
echo "[fwjoint zeroshot LIBERO-plus] CKPT=$CKPT"
echo "[fwjoint zeroshot LIBERO-plus] OUTPUT_DIR=$OUTPUT_DIR"
echo "[fwjoint zeroshot LIBERO-plus] LIBERO_PLUS_HOME=$LIBERO_PLUS_HOME"

cd "$STARVLA_DIR"

# Reads the suite size from the LIBERO-plus benchmark module. libero prints
# a huge "[info] using task orders [0,1,...,2401]" debug line on import; the
# `tail -1 | tr -dc '0-9'` strips it so bash arithmetic sees just n_tasks.
suite_size() {
  CUDA_VISIBLE_DEVICES="" "$CONDA_PY" - <<PY 2>/dev/null | tail -1 | tr -dc '0-9'
from libero.libero import benchmark
print(benchmark.get_benchmark_dict()['$1']().n_tasks)
PY
}

declare -A GPU_FOR_SUITE
GPU_FOR_SUITE[libero_spatial]="0 1"
GPU_FOR_SUITE[libero_object]="2 3"
GPU_FOR_SUITE[libero_goal]="4 5"
GPU_FOR_SUITE[libero_10]="6 7"

declare -a PIDS=()
for suite in libero_spatial libero_object libero_goal libero_10; do
  n=$(suite_size "$suite")
  half=$(( n / 2 ))
  read -r g0 g1 <<<"${GPU_FOR_SUITE[$suite]}"
  echo "[launch] $suite  total=$n  gpu${g0}:[0,${half})  gpu${g1}:[${half},${n})"

  for spec in "${g0} 0 ${half}" "${g1} ${half} ${n}"; do
    read -r gpu start end <<<"$spec"
    log="${OUTPUT_DIR}/logs/${suite}/gpu${gpu}_shard${start}_${end}.stdout"
    mkdir -p "$(dirname "$log")"
    CUDA_VISIBLE_DEVICES=${gpu} "$CONDA_PY" \
        "$STARVLA_DIR/examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_plus_lg1.py" \
        --pretrained_path "$CKPT" \
        --task_suite_name "$suite" \
        --num_trials_per_task 1 \
        --output_dir "$OUTPUT_DIR" \
        --joint_faithful True \
        --num_inference_steps ${NUM_INFERENCE_STEPS} \
        --start_idx $start --end_idx $end \
        >"$log" 2>&1 &
    PIDS+=($!)
  done
done

echo "[fwjoint zeroshot] launched ${#PIDS[@]} workers: ${PIDS[*]}"
wait "${PIDS[@]}"

echo "[fwjoint zeroshot] all workers done. Aggregating ..."
"$CONDA_PY" - <<PY
import glob, json, os
root = "${OUTPUT_DIR}"
by_suite = {}
for j in glob.glob(os.path.join(root, "logs", "*", "*_to_*.json")):
    d = json.load(open(j))
    s = d["suite"]
    bucket = by_suite.setdefault(s, {"total": 0, "ok": 0, "disturb": {}})
    bucket["total"] += d["total_episodes"]
    bucket["ok"] += d["total_successes"]
    for cat, v in d["disturb_breakdown"].items():
        b = bucket["disturb"].setdefault(cat, {"total_count": 0, "success_count": 0})
        b["total_count"] += v["total_count"]
        b["success_count"] += v["success_count"]
print("=== LIBERO-plus zeroshot (fwjoint, joint-faithful) ===")
overall_n = overall_k = 0
for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    b = by_suite.get(s)
    if not b:
        continue
    n, k = b["total"], b["ok"]
    overall_n += n; overall_k += k
    print(f"{s:18s} {k}/{n}  {100*k/max(n,1):.2f}%")
    for cat, v in sorted(b["disturb"].items()):
        tot = v["total_count"]; ok = v["success_count"]
        print(f"    {cat:24s} {ok}/{tot}  {100*ok/max(tot,1):.2f}%")
print(f"{'OVERALL':18s} {overall_k}/{overall_n}  {100*overall_k/max(overall_n,1):.2f}%")
with open(os.path.join(root, "aggregate.json"), "w") as f:
    json.dump(by_suite, f, indent=2)
PY

echo "[fwjoint zeroshot] aggregate.json saved under $OUTPUT_DIR/"
