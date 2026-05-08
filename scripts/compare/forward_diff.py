"""Strict per-tensor forward comparison between starVLA's Wan_FastWAM and
FastWAM's reference `training_loss`, given identical post-`build_inputs` tensors.

Why this script exists
----------------------
We vendored FastWAM's WanVideoDiT/ActionDiT/MoT into starVLA, then re-wired the
joint MoT path inside `Wan_FastWAM._joint_training_loss`. The vendored code is
byte-identical to upstream (we diff'd: only import lines differ), so given the
same weights the per-block math should match exactly. This harness *empirically*
verifies that — and surfaces any wrapper-level / ckpt-remapping divergence.

Strategy
--------
1. Build a deterministic fixture in latent space (skip VAE + text encoder so we
   isolate the model path). Shapes match real FastWAM LIBERO training:
   `input_latents [1, 48, 9, 14, 14]`, `context [1, 128, 4096]`, etc.
2. Phase A: build starVLA `Wan_FastWAM`, load FastWAM official ckpt via the same
   `verify_alignment.remap_*` helpers `server_wanfastwam.py` uses. Run the
   forward path step-by-step (matching `_joint_training_loss`) with `seed=0` and
   capture every intermediate tensor.
3. Phase B: same with upstream `fastwam.models.wan22.fastwam.FastWAM`. Same
   seed. Capture matching intermediates.
4. Diff each pair of tensors → max-abs-diff table. Anything > 1e-3 (bf16 noise
   floor) flags a bug.

Run:
    python scripts/compare/forward_diff.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FASTWAM_SRC = "/data/LFT-W02_data/junjie/VLA_WM/FastWAM/src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, FASTWAM_SRC)

WAN_PATH = "/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers"
FASTWAM_CKPT = (
    "/data/LFT-W02_data/junjie/VLA_WM/FastWAM/checkpoints/fastwam_release/"
    "fastwam/libero_uncond_2cam224.pt"
)
DEVICE = torch.device("cuda:1")
DTYPE = torch.bfloat16
SEED = 0
FIXTURE_PATH = "/tmp/fwdiff_fixture.pt"
STARVLA_OUT = "/tmp/fwdiff_starvla.pt"
FASTWAM_OUT = "/tmp/fwdiff_fastwam.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────
def make_fixture():
    """Deterministic post-build_inputs tensors. Shapes match LIBERO 224×224 / 33-frame."""
    B = 1
    g = torch.Generator().manual_seed(42)
    fixture = {
        # Wan2.2 TI2V-5B VAE: C=48, 4× temporal, 16× spatial. T=33 → T_lat=9, 224 → 14.
        "input_latents": (torch.randn(B, 48, 9, 14, 14, generator=g) * 0.3).to(DTYPE),
        # UMT5-XXL output: 4096-dim, padded to 128.
        "context": (torch.randn(B, 128, 4096, generator=g) * 0.05).to(DTYPE),
        "context_mask": torch.cat(
            [torch.ones(B, 9, dtype=torch.bool), torch.zeros(B, 119, dtype=torch.bool)], dim=1
        ),
        # 32-step action chunk, 7-DoF.
        "action": (torch.randn(B, 32, 7, generator=g) * 0.5).to(DTYPE),
        "action_is_pad": torch.zeros(B, 32, dtype=torch.bool),
        "image_is_pad": torch.zeros(B, 33, dtype=torch.bool),
        # Proprio (8-dim LIBERO state, normalized in [-1, 1]).
        "proprio": (torch.randn(B, 1, 8, generator=g) * 0.3).to(DTYPE),
    }
    torch.save(fixture, FIXTURE_PATH)
    print(f"[fixture] saved to {FIXTURE_PATH}")
    for k, v in fixture.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s} shape={tuple(v.shape)} dtype={v.dtype}")
    return fixture


def to_dev(d):
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Phase A: starVLA
# ─────────────────────────────────────────────────────────────────────────────
def run_starvla(fixture):
    """Reproduce `Wan_FastWAM._joint_training_loss` step-by-step on starVLA's
    components. Loads FastWAM ckpt via the same remap helpers the deployment
    server uses. Captures every intermediate."""
    import logging
    logging.basicConfig(level=logging.WARNING)

    from omegaconf import OmegaConf
    from starVLA.model.modules.world_model.wan_video_dit import WanVideoDiT
    from starVLA.model.modules.world_model.mot import MoT
    from starVLA.model.modules.action_model.FastWAM_ActionDiT import ActionDiT
    from starVLA.model.modules.action_model.FastWAM_Scheduler import (
        WanContinuousFlowMatchScheduler,
    )
    from starVLA.model.modules.world_model.FastWAM_MaskUtils import build_mot_attention_mask
    from verify_alignment import (
        remap_fastwam_action_to_ours,
        remap_fastwam_video_to_wanvideo,
    )

    print("\n[starVLA] building components...")
    t0 = time.time()
    # Build vendored WanVideoDiT (bare — no diffusers wrapper, mirrors what
    # `Wan2.py`'s `transformer` attribute holds).
    video_dit = WanVideoDiT(
        hidden_dim=3072,
        in_dim=48,
        ffn_dim=14336,
        out_dim=48,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        action_dim=7,
        action_group_causal_mask_mode="group_diagonal",
        video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=False,
    )
    action_dit = ActionDiT(
        hidden_dim=1024,
        action_dim=7,
        ffn_dim=4096,
        text_dim=4096,
        freq_dim=256,
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
        eps=1e-6,
    )
    # MoT — uses the existing video and action experts as its mixtures.
    mot = MoT(mixtures={"video": video_dit, "action": action_dit})
    print(f"[starVLA] built in {time.time()-t0:.1f}s")

    # Load FastWAM ckpt
    print("[starVLA] loading FastWAM ckpt...")
    payload = torch.load(FASTWAM_CKPT, map_location="cpu", weights_only=False)
    mot_state = payload["mot"]
    # video: remap FastWAM's mixtures.video.* to starVLA's vendored WanVideoDiT layout
    video_state, unmapped = remap_fastwam_video_to_wanvideo(mot_state)
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped video keys, e.g. {unmapped[:3]}")
    missing, unexpected = video_dit.load_state_dict(video_state, strict=False)
    print(f"  video_dit: missing={len(missing)} unexpected={len(unexpected)}")
    # action: remap FastWAM's mixtures.action.* to starVLA's ActionDiT layout
    action_state = remap_fastwam_action_to_ours(mot_state)
    missing, unexpected = action_dit.load_state_dict(action_state, strict=False)
    print(f"  action_dit: missing={len(missing)} unexpected={len(unexpected)}")
    # proprio_encoder is separate
    proprio_state = payload.get("proprio_encoder", {})
    if proprio_state:
        proprio_encoder = torch.nn.Linear(8, 4096)
        proprio_encoder.load_state_dict(proprio_state, strict=True)
        proprio_encoder = proprio_encoder.to(DEVICE, DTYPE).eval()
    else:
        proprio_encoder = None

    # Move all to device
    video_dit = video_dit.to(DEVICE, DTYPE).eval()
    action_dit = action_dit.to(DEVICE, DTYPE).eval()
    mot = mot.to(DEVICE, DTYPE).eval()

    # Build schedulers
    sched_v = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    sched_a = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    # === Run forward ===
    print("[starVLA] running forward step-by-step...")
    fx = to_dev(fixture)
    clean_latents = fx["input_latents"]
    actions_target = fx["action"]
    ctx_text = fx["context"]
    ctx_mask = fx["context_mask"]
    action_is_pad = fx["action_is_pad"]
    image_is_pad = fx["image_is_pad"]
    proprio = fx["proprio"]

    # Append proprio to context (matches FastWAM's _append_proprio_to_context)
    if proprio_encoder is not None:
        proprio_emb = proprio_encoder(proprio[:, 0, :].to(DTYPE)).unsqueeze(1)
        ctx = torch.cat([ctx_text, proprio_emb], dim=1)
        proprio_mask = torch.ones(ctx_mask.shape[0], 1, dtype=torch.bool, device=DEVICE)
        ctx_mask = torch.cat([ctx_mask, proprio_mask], dim=1)
    else:
        ctx = ctx_text

    B = clean_latents.shape[0]
    torch.manual_seed(SEED)

    # Noise + timestep (same RNG sequence as both _joint_training_loss and training_loss)
    noise_video = torch.randn_like(clean_latents)
    t_video = sched_v.sample_training_t(B, DEVICE, clean_latents.dtype)
    noisy_video = sched_v.add_noise(clean_latents, noise_video, t_video)
    target_video = sched_v.training_target(clean_latents, noise_video, t_video)
    first_frame_latents = clean_latents[:, :, 0:1].clone()
    noisy_video[:, :, 0:1] = first_frame_latents

    noise_action = torch.randn_like(actions_target)
    t_action = sched_a.sample_training_t(B, DEVICE, actions_target.dtype)
    noisy_action = sched_a.add_noise(actions_target, noise_action, t_action)
    target_action = sched_a.training_target(actions_target, noise_action, t_action)

    # pre_dit
    with torch.no_grad():
        video_pre = video_dit.pre_dit(
            x=noisy_video,
            timestep=t_video,
            context=ctx,
            context_mask=ctx_mask,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = action_dit.pre_dit(
            action_tokens=noisy_action,
            timestep=t_action,
            context=ctx,
            context_mask=ctx_mask,
        )

        # MoT mask
        Sv = video_pre["tokens"].shape[1]
        Sa = action_pre["tokens"].shape[1]
        tpf = int(video_pre["meta"]["tokens_per_frame"])
        mot_mask = build_mot_attention_mask(
            video_seq_len=Sv,
            action_seq_len=Sa,
            video_tokens_per_frame=tpf,
            video_attention_mask_mode="first_frame_causal",
            device=DEVICE,
        )

        # MoT forward
        mot_out = mot.forward(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=mot_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )

        pred_video = video_dit.post_dit(mot_out["video"], video_pre)
        pred_action = action_dit.post_dit(mot_out["action"], action_pre)

    # Loss (excluding first frame for video)
    pred_v = pred_video[:, :, 1:].float()
    target_v = target_video[:, :, 1:].float()
    video_loss_token = torch.nn.functional.mse_loss(pred_v, target_v, reduction="none").mean(dim=(1, 3, 4))
    video_loss_per_sample = video_loss_token.mean(dim=1)
    video_weight = sched_v.training_weight(t_video).to(video_loss_per_sample)
    loss_video = (video_loss_per_sample * video_weight).mean()

    action_loss_token = torch.nn.functional.mse_loss(
        pred_action.float(), target_action.float(), reduction="none"
    ).mean(dim=2)
    action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = sched_a.training_weight(t_action).to(action_loss_per_sample)
    loss_action = (action_loss_per_sample * action_weight).mean()

    out = {
        "noise_video": noise_video.detach().cpu(),
        "t_video": t_video.detach().cpu(),
        "noisy_video": noisy_video.detach().cpu(),
        "target_video": target_video.detach().cpu(),
        "noise_action": noise_action.detach().cpu(),
        "t_action": t_action.detach().cpu(),
        "noisy_action": noisy_action.detach().cpu(),
        "target_action": target_action.detach().cpu(),
        "ctx_with_proprio": ctx.detach().cpu(),
        "ctx_mask_with_proprio": ctx_mask.detach().cpu(),
        "video_pre_tokens": video_pre["tokens"].detach().cpu(),
        "video_pre_freqs": video_pre["freqs"].detach().cpu(),
        "video_pre_t_mod": video_pre["t_mod"].detach().cpu(),
        "video_pre_context": video_pre["context"].detach().cpu(),
        "action_pre_tokens": action_pre["tokens"].detach().cpu(),
        "action_pre_freqs": action_pre["freqs"].detach().cpu(),
        "action_pre_t_mod": action_pre["t_mod"].detach().cpu(),
        "action_pre_context": action_pre["context"].detach().cpu(),
        "mot_out_video": mot_out["video"].detach().cpu(),
        "mot_out_action": mot_out["action"].detach().cpu(),
        "pred_video": pred_video.detach().cpu(),
        "pred_action": pred_action.detach().cpu(),
        "loss_video": loss_video.detach().cpu(),
        "loss_action": loss_action.detach().cpu(),
    }
    torch.save(out, STARVLA_OUT)
    print(f"[starVLA] saved {len(out)} tensors to {STARVLA_OUT}")
    print(f"  loss_video={float(loss_video):.6f} loss_action={float(loss_action):.6f}")

    # Free GPU memory before phase B
    del video_dit, action_dit, mot, proprio_encoder
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Phase B: FastWAM upstream
# ─────────────────────────────────────────────────────────────────────────────
def run_fastwam(fixture):
    """Same forward path on upstream FastWAM components, same seed."""
    print("\n[FastWAM] building components...")
    t0 = time.time()
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.schedulers.scheduler_continuous import (
        WanContinuousFlowMatchScheduler,
    )

    video_dit = WanVideoDiT(
        hidden_dim=3072,
        in_dim=48,
        ffn_dim=14336,
        out_dim=48,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        action_dim=7,
        action_group_causal_mask_mode="group_diagonal",
        video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=False,
    )
    action_dit = ActionDiT(
        hidden_dim=1024,
        action_dim=7,
        ffn_dim=4096,
        text_dim=4096,
        freq_dim=256,
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
        eps=1e-6,
    )
    mot = MoT(mixtures={"video": video_dit, "action": action_dit})
    print(f"[FastWAM] built in {time.time()-t0:.1f}s")

    # Load ckpt — FastWAM stores under `mot` key with `mixtures.video.*` and `mixtures.action.*`
    print("[FastWAM] loading ckpt...")
    payload = torch.load(FASTWAM_CKPT, map_location="cpu", weights_only=False)
    mot_state = payload["mot"]
    missing, unexpected = mot.load_state_dict(mot_state, strict=False)
    print(f"  mot: missing={len(missing)} unexpected={len(unexpected)}")
    # Proprio: use FastWAM's expected layout
    proprio_state = payload.get("proprio_encoder", {})
    if proprio_state:
        proprio_encoder = torch.nn.Linear(8, 4096)
        proprio_encoder.load_state_dict(proprio_state, strict=True)
        proprio_encoder = proprio_encoder.to(DEVICE, DTYPE).eval()
    else:
        proprio_encoder = None

    video_dit = video_dit.to(DEVICE, DTYPE).eval()
    action_dit = action_dit.to(DEVICE, DTYPE).eval()
    mot = mot.to(DEVICE, DTYPE).eval()

    sched_v = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    sched_a = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    # === Run forward ===
    print("[FastWAM] running forward step-by-step...")
    fx = to_dev(fixture)
    input_latents = fx["input_latents"]
    action = fx["action"]
    ctx_text = fx["context"]
    ctx_mask = fx["context_mask"]
    proprio = fx["proprio"]

    # FastWAM's _append_proprio_to_context: same math as starVLA
    if proprio_encoder is not None:
        proprio_emb = proprio_encoder(proprio[:, 0, :].to(DTYPE)).unsqueeze(1)
        context = torch.cat([ctx_text, proprio_emb], dim=1)
        proprio_mask = torch.ones(ctx_mask.shape[0], 1, dtype=torch.bool, device=DEVICE)
        context_mask = torch.cat([ctx_mask, proprio_mask], dim=1)
    else:
        context = ctx_text
        context_mask = ctx_mask

    B = input_latents.shape[0]
    torch.manual_seed(SEED)

    noise_video = torch.randn_like(input_latents)
    timestep_video = sched_v.sample_training_t(B, DEVICE, input_latents.dtype)
    latents = sched_v.add_noise(input_latents, noise_video, timestep_video)
    target_video = sched_v.training_target(input_latents, noise_video, timestep_video)
    first_frame_latents = input_latents[:, :, 0:1]
    latents[:, :, 0:1] = first_frame_latents

    noise_action = torch.randn_like(action)
    timestep_action = sched_a.sample_training_t(B, DEVICE, action.dtype)
    noisy_action = sched_a.add_noise(action, noise_action, timestep_action)
    target_action = sched_a.training_target(action, noise_action, timestep_action)

    with torch.no_grad():
        video_pre = video_dit.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = action_dit.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        # Replicate FastWAM's `_build_mot_attention_mask` inline (it depends on
        # `self.video_expert.build_video_to_video_mask`, so calling it as a static
        # function won't work).
        Sv = video_pre["tokens"].shape[1]
        Sa = action_pre["tokens"].shape[1]
        tpf = int(video_pre["meta"]["tokens_per_frame"])
        total = Sv + Sa
        mot_mask = torch.zeros((total, total), dtype=torch.bool, device=DEVICE)
        mot_mask[:Sv, :Sv] = video_dit.build_video_to_video_mask(
            video_seq_len=Sv, video_tokens_per_frame=tpf, device=DEVICE,
        )
        mot_mask[Sv:, Sv:] = True
        ff_tokens = min(tpf, Sv)
        mot_mask[Sv:, :ff_tokens] = True

        mot_out = mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=mot_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )

        pred_video = video_dit.post_dit(mot_out["video"], video_pre)
        pred_action = action_dit.post_dit(mot_out["action"], action_pre)

    pred_v = pred_video[:, :, 1:].float()
    target_v = target_video[:, :, 1:].float()
    video_loss_token = torch.nn.functional.mse_loss(pred_v, target_v, reduction="none").mean(dim=(1, 3, 4))
    video_loss_per_sample = video_loss_token.mean(dim=1)
    video_weight = sched_v.training_weight(timestep_video).to(video_loss_per_sample)
    loss_video = (video_loss_per_sample * video_weight).mean()

    action_loss_token = torch.nn.functional.mse_loss(
        pred_action.float(), target_action.float(), reduction="none"
    ).mean(dim=2)
    action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = sched_a.training_weight(timestep_action).to(action_loss_per_sample)
    loss_action = (action_loss_per_sample * action_weight).mean()

    out = {
        "noise_video": noise_video.detach().cpu(),
        "t_video": timestep_video.detach().cpu(),
        "noisy_video": latents.detach().cpu(),
        "target_video": target_video.detach().cpu(),
        "noise_action": noise_action.detach().cpu(),
        "t_action": timestep_action.detach().cpu(),
        "noisy_action": noisy_action.detach().cpu(),
        "target_action": target_action.detach().cpu(),
        "ctx_with_proprio": context.detach().cpu(),
        "ctx_mask_with_proprio": context_mask.detach().cpu(),
        "video_pre_tokens": video_pre["tokens"].detach().cpu(),
        "video_pre_freqs": video_pre["freqs"].detach().cpu(),
        "video_pre_t_mod": video_pre["t_mod"].detach().cpu(),
        "video_pre_context": video_pre["context"].detach().cpu(),
        "action_pre_tokens": action_pre["tokens"].detach().cpu(),
        "action_pre_freqs": action_pre["freqs"].detach().cpu(),
        "action_pre_t_mod": action_pre["t_mod"].detach().cpu(),
        "action_pre_context": action_pre["context"].detach().cpu(),
        "mot_out_video": mot_out["video"].detach().cpu(),
        "mot_out_action": mot_out["action"].detach().cpu(),
        "pred_video": pred_video.detach().cpu(),
        "pred_action": pred_action.detach().cpu(),
        "loss_video": loss_video.detach().cpu(),
        "loss_action": loss_action.detach().cpu(),
    }
    torch.save(out, FASTWAM_OUT)
    print(f"[FastWAM] saved {len(out)} tensors to {FASTWAM_OUT}")
    print(f"  loss_video={float(loss_video):.6f} loss_action={float(loss_action):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Diff
# ─────────────────────────────────────────────────────────────────────────────
def diff_outputs():
    a = torch.load(STARVLA_OUT, map_location="cpu")
    b = torch.load(FASTWAM_OUT, map_location="cpu")
    keys = sorted(set(a.keys()) | set(b.keys()))
    BF16_TOL = 1e-3
    print("\n" + "=" * 78)
    print(f"{'tensor':30s} {'starVLA shape':22s} {'max-abs-diff':>14s}  status")
    print("=" * 78)
    for k in keys:
        if k not in a:
            print(f"  {k}: missing on starVLA side")
            continue
        if k not in b:
            print(f"  {k}: missing on FastWAM side")
            continue
        ta, tb = a[k], b[k]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            print(f"  {k}: non-tensor")
            continue
        if ta.shape != tb.shape:
            print(f"  {k}: SHAPE MISMATCH {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        if ta.dtype != tb.dtype:
            ta = ta.float()
            tb = tb.float()
        diff = (ta.float() - tb.float()).abs().max().item()
        status = "✓" if diff < BF16_TOL else ("≈" if diff < 1e-2 else "✗")
        print(f"  {k:30s} {str(tuple(ta.shape)):22s} {diff:14.6e}  {status}")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fixture = make_fixture()
    run_starvla(fixture)
    run_fastwam(fixture)
    diff_outputs()
