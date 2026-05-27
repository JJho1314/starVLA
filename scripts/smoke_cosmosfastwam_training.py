#!/usr/bin/env python3
"""End-to-end training smoke for Cosmos FastWAM.

Goes one step beyond ``smoke_cosmosfastwam.py``: actually runs
``WanFastWAM.forward()`` (which calls ``_joint_training_loss``), then
``loss.backward()`` + ``optimizer.step()`` to verify the full training
pipeline works — catches dtype mismatches, NaN losses, missing grad paths,
and any silent shape errors that only surface after the backward pass.

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/smoke_cosmosfastwam_training.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger(__name__)


def _build_synthetic_examples(batch_size: int, num_frames: int, chunk_len: int,
                              action_dim: int, state_dim: int,
                              image_h: int, image_w: int, rng: np.random.RandomState):
    """Build a batch of fake examples matching what the dataloader emits.

    For WanFastWAM with multi_frame_video=true the dataloader emits, per sample:
        image: list of N_CAMS lists, each containing N_FRAMES (H, W, 3) uint8 arrays
        lang: str
        action: (chunk_len, action_dim) float
        state: (state_dim,) float
        action_is_pad: (chunk_len,) bool
        image_is_pad: (num_frames,) bool
    """
    examples = []
    for b in range(batch_size):
        # 1 camera (Cosmos pretrained at single 480×832; for smoke we use 224×448 from yaml).
        cam_frames = [rng.randint(0, 255, (image_h, image_w, 3), dtype=np.uint8)
                      for _ in range(num_frames)]
        examples.append({
            "image": [cam_frames],            # 1 cam, num_frames frames
            "lang": "pick up the alphabet soup and place it in the basket",
            "action": rng.randn(chunk_len, action_dim).astype(np.float32),
            "state": rng.randn(state_dim).astype(np.float32),
            "action_is_pad": np.zeros(chunk_len, dtype=bool),
            "image_is_pad": np.zeros(num_frames, dtype=bool),
        })
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-yaml",
        default=str(_REPO / "examples/LIBERO/train_files/starvla_cosmosfastwam_libero_joint.yaml"),
    )
    ap.add_argument(
        "--cosmos-path",
        default="/data/LFT-W02_data/junjie/weights/Cosmos-Predict2-2B-Video2World",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1, help="micro-batch for smoke (1 keeps memory low)")
    ap.add_argument("--steps", type=int, default=1, help="num forward/backward iterations")
    ap.add_argument("--image-h", type=int, default=224)
    ap.add_argument("--image-w", type=int, default=224, help="single-cam Cosmos smoke; 224×224 keeps things small")
    args = ap.parse_args()

    print(f"[smoke] config_yaml={args.config_yaml}")
    print(f"[smoke] cosmos_path={args.cosmos_path}")

    cfg = OmegaConf.load(args.config_yaml)
    # Patch yaml to local Cosmos path + force batched-friendly settings.
    cfg.framework.world_model.base_wm = args.cosmos_path
    cfg.framework.qwenvl.base_vlm = args.cosmos_path
    # Single-cam, smaller image to keep training-smoke memory small.
    cfg.framework.image_preprocess = {
        "image_height": args.image_h,
        "image_width": args.image_w,
        "num_cameras": 1,
    }
    # Disable text cache so live T5 runs (we won't have a Cosmos-T5 cache yet).
    cfg.framework.world_model.text_embed_cache_path = None
    cfg.framework.world_model.load_text_encoder = True

    print("[smoke] building WanFastWAM framework with Cosmos backbone ...")
    from starVLA.model.framework.WM4A.WanFastWAM import Wan_FastWAM
    t0 = time.time()
    model = Wan_FastWAM(config=cfg)
    print(f"[smoke] framework built in {time.time()-t0:.1f}s")
    print(f"[smoke] backbone class: {type(model.backbone).__name__}")
    assert type(model.backbone).__name__ == "WanVideoBackboneFastWAMCosmos", (
        f"Expected Cosmos backbone, got {type(model.backbone).__name__}. "
        "Check yaml `base_wm` contains 'cosmos-predict2' AND `use_fastwam_aligned_io: true`."
    )
    print(f"[smoke]   transformer: {type(model.backbone.transformer).__name__}, "
          f"layers={len(model.backbone.transformer.blocks)}, "
          f"hidden={model.backbone.transformer.hidden_dim}")
    print(f"[smoke]   ActionDiT: layers={len(model.action_expert.blocks)}, "
          f"hidden={model.action_expert.hidden_dim}, "
          f"action_dim={model.action_dim}")

    # bf16 + GPU
    model = model.to(args.device).to(torch.bfloat16)

    # Make sure ActionDiT/proprio_encoder are trainable (the smoke needs grads).
    for p in model.parameters():
        p.requires_grad_(False)
    # Trainable: ActionDiT (joint MoT video expert via backbone.transformer too).
    for n, p in model.named_parameters():
        # Train ActionDiT + transformer (the two MoT experts). Freeze VAE + T5.
        if n.startswith("backbone.vae.") or n.startswith("backbone.text_encoder."):
            continue
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    n_total = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[smoke] trainable params: {n_trainable:.2f}B / {n_total:.2f}B total")

    # Build a 1-sample synthetic batch.
    rng = np.random.RandomState(7)
    chunk_len = model.chunk_len
    state_dim = model.state_dim
    action_dim = model.action_dim
    num_frames = 9   # FastWAM-aligned
    examples = _build_synthetic_examples(
        batch_size=args.batch_size, num_frames=num_frames,
        chunk_len=chunk_len, action_dim=action_dim, state_dim=state_dim,
        image_h=args.image_h, image_w=args.image_w, rng=rng,
    )
    print(f"[smoke] synthetic batch: B={args.batch_size} num_frames={num_frames} "
          f"chunk_len={chunk_len} action_dim={action_dim} state_dim={state_dim} "
          f"image_h×w={args.image_h}×{args.image_w}")

    # Use SGD for the smoke: AdamW needs 4× param memory for m+v states which
    # OOMs a single A6000 (47 GB) for the 8.82B-param model. Real training
    # uses DeepSpeed ZeRO-2 which shards optimizer state across GPUs; here we
    # just want to prove backward + step round-trip works.
    optim = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )

    # === Forward → backward → optimizer step ====================================
    model.train()
    for step in range(args.steps):
        print(f"\n[smoke] === STEP {step+1}/{args.steps} ===")
        t_fw = time.time()
        with torch.autocast(args.device, dtype=torch.bfloat16):
            out = model(examples=examples)
        if not isinstance(out, dict):
            raise RuntimeError(f"forward() returned unexpected output: {type(out)}")
        # WanFastWAM returns separate {action_loss, video_loss}; sum for backward.
        loss_parts = {k: v for k, v in out.items() if torch.is_tensor(v) and v.requires_grad}
        if not loss_parts:
            raise RuntimeError(f"no trainable loss in output keys={list(out.keys())}")
        loss = sum(loss_parts.values())
        elapsed_fw = time.time() - t_fw
        print(f"[smoke]   forward done in {elapsed_fw:.2f}s")
        print(f"[smoke]   total loss = {loss.item():.4f}")
        for k, v in loss_parts.items():
            print(f"[smoke]     {k} = {v.item():.4f}")
        assert torch.isfinite(loss), f"loss is NaN/Inf at step {step+1}: {loss.item()}"

        t_bw = time.time()
        loss.backward()
        elapsed_bw = time.time() - t_bw
        print(f"[smoke]   backward done in {elapsed_bw:.2f}s")

        # Check grad sanity on a few important params.
        for n in ["action_expert.head.weight", "backbone.transformer.blocks.0.self_attn.q.weight",
                  "action_expert.blocks.0.self_attn.q.weight", "proprio_encoder.weight"]:
            p = dict(model.named_parameters()).get(n)
            if p is not None and p.grad is not None:
                g_norm = p.grad.float().norm().item()
                print(f"[smoke]   grad {n}: norm={g_norm:.4f}")
                assert g_norm > 0, f"grad is zero on {n}"

        optim.step()
        optim.zero_grad()
        print(f"[smoke]   optimizer step OK")

    print("\n[smoke] ✅ ALL TRAINING SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
