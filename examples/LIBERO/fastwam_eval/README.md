# FastWAM-style LIBERO Evaluation (in-tree copy)

Verbatim copy of FastWAM's official eval pipeline, dropped into starVLA so it can be
launched without bouncing between repos. Useful as the **ground-truth reference**:
runs `yuanty/fastwam` released ckpt through the same eval logic the paper used.

## Files

| File | Source | Purpose |
|---|---|---|
| `run_libero_manager.py` | FastWAM `experiments/libero/run_libero_manager.py` | Hydra entry; spawns workers |
| `run_libero_parallel_test.sh` | FastWAM same dir | Per-GPU worker launcher |
| `eval_libero_single.py` | FastWAM same dir | One task, one process — the meat |
| `libero_utils.py` / `action_ensembler.py` / `summarize_results.py` | FastWAM same dir | Helpers |
| `configs/sim_libero.yaml` + `configs/{train,task,data,model}/...` | FastWAM `configs/` | Hydra config tree |

The only edits vs. upstream FastWAM:
- `config_path="../../configs"` → `config_path="./configs"` (so configs resolve under this dir).
- Subprocess paths inside the `.sh` retargeted to `examples/LIBERO/fastwam_eval/...`.

## Prerequisites

1. **Conda env** with FastWAM: install per FastWAM `README.md` — `fastwam` env that has the FastWAM package
   (`pip install -e .` from FastWAM repo) **plus** `mujoco==3.3.2` and the LIBERO sim env
   (clone https://github.com/Lifelong-Robot-Learning/LIBERO and `pip install -e .`).

2. **Wan2.2-TI2V-5B raw weights** at `./checkpoints/Wan-AI/Wan2.2-TI2V-5B`. Symlink
   the modelscope-style download here (the dir that contains `diffusion_pytorch_model-*.safetensors`,
   `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.2_VAE.pth`, etc.).

3. **ActionDiT preprocessed file** at `./checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`:
   ```bash
   # in FastWAM repo:
   python scripts/preprocess_action_dit_backbone.py \
     --model-config configs/model/fastwam.yaml \
     --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
     --device cuda --dtype bfloat16
   # then symlink the result to starVLA's ./checkpoints/
   ```

4. **Released ckpt + dataset stats** at `./checkpoints/fastwam_release/`:
   ```bash
   huggingface-cli download yuanty/fastwam \
     libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
     --local-dir ./checkpoints/fastwam_release
   ```

## Run

```bash
# from starVLA repo root
bash examples/LIBERO/fastwam_eval/run_eval.sh
```

Default: 1 GPU × 50 trials × 4 suites (spatial/object/goal/10). Override via env:

```bash
NUM_GPUS=4 NUM_TRIALS=50 SUITES="libero_object" \
  bash examples/LIBERO/fastwam_eval/run_eval.sh
```

Outputs land under `playground/eval_logs/<timestamp>_fastwam_eval/<suite>/`,
with FastWAM's standard per-task `_summary.json` files and a final aggregate.

## Why have this here?

1. **Ground-truth SR**: confirms what the released ckpt scores under FastWAM's own setup.
2. **A/B with starVLA's eval pipeline** (`examples/LIBERO/eval_files/`): isolates whether
   any SR gap is in the model itself vs. the eval plumbing.
