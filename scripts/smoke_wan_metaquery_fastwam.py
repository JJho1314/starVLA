"""Single-GPU smoke test for WanMetaQueryFastWAM.

Goal: build the model end-to-end with real weights and run one forward
pass on a synthesized batch (no real dataloader). This flushes out the
most likely failure modes:

    * monkey-patch of `backbone.build_inputs`
    * Qwen3-VL tokenizer resize + row-mask hook
    * concat(text_embeds, vlm_ctx) shape & dtype
    * connector forward + RMSNorm
    * joint MoT forward with the augmented context

Run with:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_wan_metaquery_fastwam.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.base_framework import build_framework


def make_dummy_batch(num_frames: int, chunk_len: int):
    """Build one VLA example that matches WanFastWAM.forward()'s expected
    schema (see `forward` in WanFastWAM.py for the key list)."""
    # multi_frame_video=True → image is List-of-cams, each cam is list of T PIL frames.
    H, W = 224, 224
    cam_pri = [Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
               for _ in range(num_frames)]
    cam_wri = [Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
               for _ in range(num_frames)]
    example = {
        "image": [cam_pri, cam_wri],                       # 2 cams, T frames each
        "lang": "pick up the red block and place it in the basket",
        "action": np.random.randn(chunk_len, 7).astype(np.float32),
        "state": np.zeros((8,), dtype=np.float32),
        "action_is_pad": np.zeros((chunk_len,), dtype=bool),
        "image_is_pad": np.zeros((num_frames,), dtype=bool),
    }
    return [example]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero_local.yaml",
    )
    parser.add_argument("--backward", action="store_true", help="run backward (much higher memory)")
    args = parser.parse_args()

    print(f"[smoke] loading config: {args.config}")
    cfg = OmegaConf.load(args.config)

    print(f"[smoke] cuda devices visible: {torch.cuda.device_count()}, "
          f"using device 0 = {torch.cuda.get_device_name(0)}")
    print(f"[smoke] mem before build: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"[smoke] building framework {cfg.framework.name} …")
    model = build_framework(cfg)
    print(f"[smoke] framework class: {type(model).__name__}")

    # Cast model to bf16 and move to GPU. We skip Adam etc; smoke test only.
    model = model.to(dtype=torch.bfloat16, device="cuda")
    model.eval() if not args.backward else model.train()

    n_total = sum(p.numel() for p in model.parameters()) / 1e9
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    print(f"[smoke] params: total={n_total:.2f}B  trainable={n_train:.2f}B")
    print(f"[smoke] mem after build+cast: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    chunk_len = model.chunk_len
    # Wan's video pipeline expects ≥2 frames latent-wise; FastWAM trains with 9 video frames.
    num_frames = 9
    batch = make_dummy_batch(num_frames, chunk_len)
    print(f"[smoke] dummy batch: 1 sample, T={num_frames} frames × 2 cams, chunk_len={chunk_len}")

    if args.backward:
        out = model(batch)
    else:
        with torch.no_grad():
            out = model(batch)
    print(f"[smoke] forward OK — losses: " + ", ".join(
        f"{k}={float(v):.4f}" for k, v in out.items() if isinstance(v, torch.Tensor)
    ))
    print(f"[smoke] peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if args.backward:
        total = sum(v for k, v in out.items() if isinstance(v, torch.Tensor))
        total.backward()
        print(f"[smoke] backward OK — final mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
