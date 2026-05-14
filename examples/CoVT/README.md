# CoVT — Chain-of-Visual-Thought (segmentation-only) on Qwen3-VL-4B

A starVLA implementation of [CoVT](https://arxiv.org/abs/2511.19418), trimmed
to a single visual anchor (segmentation) and wired up to the
[RoboInter-Data](https://huggingface.co/datasets/InternRobotics/RoboInter-Data)
DROID + RH20T annotations.

The model learns to emit a `<think> ... <|sam_pad|>×8 ... </think>` block as
visual chain-of-thought before answering with `primitive_skill` and
`subtask`. The 8 sam-pad token hidden states are projected to SAM's prompt
embedding dimension and decoded by SAM's frozen mask decoder; the predicted
masks are matched against the dataset's GT masks via Hungarian assignment.

## File map

| File | Purpose |
|------|---------|
| `starVLA/model/framework/VLM4A/QwenCoVT.py` | framework: VLM + sam projection / query / cross-attn |
| `starVLA/model/modules/visual_anchor/sam_anchor.py` | frozen SAM teacher (image_embed + mask_decoder) |
| `starVLA/model/modules/visual_anchor/losses.py` | dice / focal / Hungarian-match seg loss |
| `starVLA/dataloader/robointer_covt/` | LMDB + sam_mask npz + video → tokenized samples |
| `starVLA/training/train_starcovt.py` | training entry — dispatches to `model.forward_vlm` |

## Prerequisites

```bash
# SAM + lmdb + decord
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install lmdb decord

# Download the SAM ViT-H checkpoint
mkdir -p playground/Pretrained_models
wget -O playground/Pretrained_models/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Qwen3-VL-4B (will be auto-downloaded if you skip this)
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

## Run

```bash
bash examples/CoVT/train_files/run_train_covt.sh
```

## Stage scheduling

`framework.covt.stage_*_steps` in the YAML control the curriculum:

```
step ≤ stage_warmup_steps  : LM only (no anchor in prompt)
step ≤ stage_align_steps   : feature alignment only (cheap, no SAM decode)
step ≤ stage_full_steps    : full mask reconstruction loss (Hungarian-match)
step  > stage_full_steps   : LM only (efficient inference)
```

Default starts at `align` for 500 steps then jumps to `full` for the rest.

## Adding more anchors (depth / dino / edge / …)

1. Add a teacher under `starVLA/model/modules/visual_anchor/` (mirroring
   `sam_anchor.py`'s `encode_image` / `decode_with_tokens` interface).
2. Register it in `visual_anchor/__init__.py:build_anchors`.
3. Add `<|<name>_pad|>` to `NEW_SPECIAL_TOKENS` in `QwenCoVT.py` and a
   matching projection / query / cross-attn / loss-call in
   `QwenCoVT.forward_vlm`.

## Notes

- The frozen SAM ViT-H teacher costs ≈ 2.5 GB GPU memory; if you're tight
  on VRAM, swap to `vit_b` / `vit_l` or move the teacher to CPU.
- `RoboInterCoVTDataset` only keeps episodes that have a matching
  `sam_mask.npz`; on the demo subset that's 119 / 120 episodes.
- Training scope is `forward_vlm` only — no action head — so this framework
  cannot consume `vla` batches by design (`supports_training_tag` returns
  `False` for `vla`).
