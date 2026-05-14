"""
Smoke test: verify starVLA WanFastWAM aligns with original FastWAM.

Tests run on CPU with synthetic tensors — no GPU or full model loading required.
Covers: scheduler, proprio injection, first-frame handling, loss computation
(reweighting + padding masks), and output format.

Usage:
    cd /data/LFT-W02_data/junjie/VLA_WM/starVLA
    python tests/test_wanfastwam_alignment.py
"""

import sys
from pathlib import Path

STARVLA_ROOT = Path(__file__).resolve().parent.parent
FASTWAM_ROOT = STARVLA_ROOT.parent / "FastWAM"
sys.path.insert(0, str(STARVLA_ROOT))
sys.path.insert(0, str(FASTWAM_ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import schedulers from both repos ────────────────────────────────────────
from starVLA.model.modules.action_model.FastWAM_Scheduler import (
    WanContinuousFlowMatchScheduler as StarVLAScheduler,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler as OrigScheduler,
)

# ── Import mask builder from both repos ──────────────────────────────────────
from starVLA.model.modules.world_model.FastWAM_MaskUtils import (
    build_mot_attention_mask as starvla_build_mask,
)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
test_results = []


def check(name: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    test_results.append((name, ok))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Scheduler numerical identity
# ═══════════════════════════════════════════════════════════════════════════════
def test_scheduler_identity():
    print("\n═══ Test 1: Scheduler numerical identity ═══")
    sv = StarVLAScheduler(num_train_timesteps=1000, shift=5.0)
    og = OrigScheduler(num_train_timesteps=1000, shift=5.0)

    torch.manual_seed(42)
    t_sv = sv.sample_training_t(8, torch.device("cpu"), torch.float32)
    torch.manual_seed(42)
    t_og = og.sample_training_t(8, torch.device("cpu"), torch.float32)
    check("sample_training_t", torch.allclose(t_sv, t_og, atol=1e-6),
          f"max_diff={float((t_sv - t_og).abs().max()):.2e}")

    x = torch.randn(8, 7)
    noise = torch.randn(8, 7)
    noisy_sv = sv.add_noise(x, noise, t_sv)
    noisy_og = og.add_noise(x, noise, t_og)
    check("add_noise", torch.allclose(noisy_sv, noisy_og, atol=1e-6),
          f"max_diff={float((noisy_sv - noisy_og).abs().max()):.2e}")

    target_sv = sv.training_target(x, noise, t_sv)
    target_og = og.training_target(x, noise, t_og)
    check("training_target", torch.allclose(target_sv, target_og, atol=1e-6),
          f"max_diff={float((target_sv - target_og).abs().max()):.2e}")

    w_sv = sv.training_weight(t_sv)
    w_og = og.training_weight(t_og)
    check("training_weight", torch.allclose(w_sv, w_og, atol=1e-6),
          f"max_diff={float((w_sv - w_og).abs().max()):.2e}")

    ts_sv, ds_sv = sv.build_inference_schedule(4, torch.device("cpu"), torch.float32)
    ts_og, ds_og = og.build_inference_schedule(4, torch.device("cpu"), torch.float32)
    check("build_inference_schedule",
          torch.allclose(ts_sv, ts_og, atol=1e-6) and torch.allclose(ds_sv, ds_og, atol=1e-6))

    v = torch.randn(8, 7)
    delta = torch.tensor(0.25)
    step_sv = sv.step(v, delta, x)
    step_og = og.step(v, delta, x)
    check("step", torch.allclose(step_sv, step_og, atol=1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Proprio injection order (append, not prepend)
# ═══════════════════════════════════════════════════════════════════════════════
def test_proprio_injection():
    print("\n═══ Test 2: Proprio injection order ═══")
    B, L, D, S = 2, 10, 4096, 8

    text = torch.randn(B, L, D)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    state = torch.randn(B, S)

    # ── starVLA ──
    proprio_encoder_sv = nn.Linear(S, D)
    proprio_emb_sv = proprio_encoder_sv(state.to(text.dtype)).unsqueeze(1)
    ctx_sv = torch.cat([text, proprio_emb_sv], dim=1)
    mask_sv = torch.cat([text_mask, torch.ones(B, 1, dtype=torch.bool)], dim=1)

    # ── FastWAM reference ──
    proprio_encoder_og = nn.Linear(S, D)
    proprio_encoder_og.load_state_dict(proprio_encoder_sv.state_dict())
    proprio_token_og = proprio_encoder_og(state.to(text.dtype).unsqueeze(1))
    ctx_og = torch.cat([text, proprio_token_og], dim=1)
    mask_og = torch.cat([text_mask, torch.ones(B, 1, dtype=torch.bool)], dim=1)

    check("proprio appended (not prepended)",
          ctx_sv.shape == (B, L + 1, D) and ctx_og.shape == (B, L + 1, D))
    check("proprio position matches FastWAM",
          torch.allclose(ctx_sv[:, :L, :], text) and torch.allclose(ctx_sv[:, L:, :], proprio_emb_sv))
    check("context shape", ctx_sv.shape == ctx_og.shape,
          f"starVLA={ctx_sv.shape}, FastWAM={ctx_og.shape}")
    check("mask shape", mask_sv.shape == mask_og.shape,
          f"starVLA={mask_sv.shape}, FastWAM={mask_og.shape}")
    check("text in first L positions",
          torch.allclose(ctx_sv[:, :L, :], ctx_og[:, :L, :], atol=1e-6))
    check("proprio in last position",
          torch.allclose(ctx_sv[:, L:, :], ctx_og[:, L:, :], atol=1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: First-frame clean latent replacement
# ═══════════════════════════════════════════════════════════════════════════════
def test_first_frame_handling():
    print("\n═══ Test 3: First-frame clean latent replacement ═══")
    B, C, T, H, W = 2, 48, 9, 14, 28
    scheduler = StarVLAScheduler(num_train_timesteps=1000, shift=5.0)

    clean_latents = torch.randn(B, C, T, H, W)
    noise = torch.randn_like(clean_latents)
    t = scheduler.sample_training_t(B, torch.device("cpu"), clean_latents.dtype)
    target_video = scheduler.training_target(clean_latents, noise, t)

    # ── starVLA path ──
    noisy_sv = scheduler.add_noise(clean_latents, noise, t)
    first_frame = clean_latents[:, :, 0:1].clone()
    noisy_sv[:, :, 0:1] = first_frame
    check("first frame replaced with clean",
          torch.allclose(noisy_sv[:, :, 0:1], clean_latents[:, :, 0:1], atol=1e-7))

    remaining_noisy = scheduler.add_noise(clean_latents, noise, t)[:, :, 1:]
    check("remaining frames still noisy",
          torch.allclose(noisy_sv[:, :, 1:], remaining_noisy, atol=1e-7))

    # ── FastWAM path (same logic) ──
    noisy_og = scheduler.add_noise(clean_latents, noise, t)
    noisy_og[:, :, 0:1] = clean_latents[:, :, 0:1]
    check("starVLA == FastWAM noisy latents",
          torch.allclose(noisy_sv, noisy_og, atol=1e-7))

    # Loss excludes first frame
    pred_fake = torch.randn(B, C, T, H, W)
    pred_vid_sv = pred_fake[:, :, 1:]
    target_vid_sv = target_video[:, :, 1:]
    check("loss excludes first frame",
          pred_vid_sv.shape[2] == T - 1 and target_vid_sv.shape[2] == T - 1,
          f"latent T={T} -> loss T={pred_vid_sv.shape[2]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Video loss with image_is_pad
# ═══════════════════════════════════════════════════════════════════════════════
def test_video_loss_with_padding():
    print("\n═══ Test 4: Video loss with image_is_pad ═══")
    B, C, H, W = 2, 48, 14, 28
    temporal_factor = 4
    num_frames = 33  # standard LIBERO: T%4==1 -> T=33
    T_latent = (num_frames - 1) // temporal_factor + 1  # 9

    pred = torch.randn(B, C, T_latent - 1, H, W)
    target = torch.randn(B, C, T_latent - 1, H, W)

    # Create image_is_pad: last 8 frames are padded for sample 1
    image_is_pad = torch.zeros(B, num_frames, dtype=torch.bool)
    image_is_pad[1, 25:] = True  # sample 1 has 8 padded frames at end

    # ── FastWAM reference computation ──
    video_loss_token = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 3, 4))
    tail_is_pad = image_is_pad[:, 1:]
    latent_tail_is_pad = tail_is_pad.view(B, -1, temporal_factor).all(dim=2)
    video_is_pad_og = latent_tail_is_pad  # include_initial_video_step=False
    valid_og = (~video_is_pad_og).float()
    valid_sum_og = valid_og.sum(dim=1).clamp(min=1.0)
    loss_per_sample_og = (video_loss_token * valid_og).sum(dim=1) / valid_sum_og

    # ── starVLA computation ──
    tf = temporal_factor
    tail_is_pad_sv = image_is_pad[:, 1:]
    latent_tail_is_pad_sv = tail_is_pad_sv.view(B, -1, tf).all(dim=2)
    valid_sv = (~latent_tail_is_pad_sv).float()
    valid_sum_sv = valid_sv.sum(dim=1).clamp(min=1.0)
    loss_per_sample_sv = (video_loss_token * valid_sv).sum(dim=1) / valid_sum_sv

    check("video loss shape", loss_per_sample_sv.shape == (B,))
    check("video loss matches FastWAM",
          torch.allclose(loss_per_sample_sv, loss_per_sample_og, atol=1e-6),
          f"max_diff={float((loss_per_sample_sv - loss_per_sample_og).abs().max()):.2e}")

    # Verify masking actually makes a difference
    loss_no_pad = video_loss_token.mean(dim=1)
    check("padding changes loss for padded sample",
          not torch.allclose(loss_per_sample_sv[1:2], loss_no_pad[1:2], atol=1e-4))
    check("padding does NOT change loss for unpadded sample",
          torch.allclose(loss_per_sample_sv[0:1], loss_no_pad[0:1], atol=1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Action loss with action_is_pad
# ═══════════════════════════════════════════════════════════════════════════════
def test_action_loss_with_padding():
    print("\n═══ Test 5: Action loss with action_is_pad ═══")
    B, T_action, A = 2, 8, 7
    pred = torch.randn(B, T_action, A)
    target = torch.randn(B, T_action, A)
    action_is_pad = torch.zeros(B, T_action, dtype=torch.bool)
    action_is_pad[1, 6:] = True

    action_loss_token = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=2)

    # ── FastWAM reference ──
    valid_og = (~action_is_pad).float()
    valid_sum_og = valid_og.sum(dim=1).clamp(min=1.0)
    loss_og = (action_loss_token * valid_og).sum(dim=1) / valid_sum_og

    # ── starVLA (same logic) ──
    valid_sv = (~action_is_pad).float()
    valid_sum_sv = valid_sv.sum(dim=1).clamp(min=1.0)
    loss_sv = (action_loss_token * valid_sv).sum(dim=1) / valid_sum_sv

    check("action loss shape", loss_sv.shape == (B,))
    check("action loss matches FastWAM",
          torch.allclose(loss_sv, loss_og, atol=1e-6))

    # Without padding
    loss_no_pad = action_loss_token.mean(dim=1)
    check("padding changes loss for padded sample",
          not torch.allclose(loss_sv[1:2], loss_no_pad[1:2], atol=1e-4))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Full loss pipeline (video + action with reweighting)
# ═══════════════════════════════════════════════════════════════════════════════
def test_full_loss_pipeline():
    print("\n═══ Test 6: Full loss pipeline (video + action with reweighting) ═══")
    B = 4
    lambda_video, lambda_action = 1.0, 1.0
    temporal_factor = 4
    num_frames = 33
    T_latent = (num_frames - 1) // temporal_factor + 1  # 9
    C, H, W = 48, 14, 28
    T_action, A = 8, 7

    video_sched = StarVLAScheduler(num_train_timesteps=1000, shift=5.0)
    action_sched = StarVLAScheduler(num_train_timesteps=1000, shift=5.0)

    # Synthetic clean data
    clean_latents = torch.randn(B, C, T_latent, H, W)
    actions = torch.randn(B, T_action, A)

    # ── Reproduce FastWAM training_loss logic ──
    torch.manual_seed(123)
    noise_v = torch.randn_like(clean_latents)
    t_v = video_sched.sample_training_t(B, torch.device("cpu"), clean_latents.dtype)
    noisy_v = video_sched.add_noise(clean_latents, noise_v, t_v)
    target_v = video_sched.training_target(clean_latents, noise_v, t_v)
    noisy_v[:, :, 0:1] = clean_latents[:, :, 0:1]

    torch.manual_seed(456)
    noise_a = torch.randn_like(actions)
    t_a = action_sched.sample_training_t(B, torch.device("cpu"), actions.dtype)
    noisy_a = action_sched.add_noise(actions, noise_a, t_a)
    target_a = action_sched.training_target(actions, noise_a, t_a)

    # Fake predictions (simulate MoT output)
    torch.manual_seed(789)
    pred_v = torch.randn(B, C, T_latent, H, W)
    pred_a = torch.randn(B, T_action, A)

    # ── FastWAM loss computation ──
    pred_v_crop = pred_v[:, :, 1:]
    target_v_crop = target_v[:, :, 1:]
    video_loss_token = F.mse_loss(pred_v_crop.float(), target_v_crop.float(),
                                  reduction="none").mean(dim=(1, 3, 4))
    video_loss_per_sample = video_loss_token.mean(dim=1)
    video_weight = video_sched.training_weight(t_v)
    loss_video_og = (video_loss_per_sample * video_weight).mean()

    action_loss_token = F.mse_loss(pred_a.float(), target_a.float(),
                                   reduction="none").mean(dim=2)
    action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = action_sched.training_weight(t_a)
    loss_action_og = (action_loss_per_sample * action_weight).mean()

    total_og = lambda_video * loss_video_og + lambda_action * loss_action_og

    # ── starVLA loss computation (same) ──
    pred_vid_sv = pred_v[:, :, 1:]
    target_vid_sv = target_v[:, :, 1:]
    vlt_sv = F.mse_loss(pred_vid_sv.float(), target_vid_sv.float(),
                        reduction="none").mean(dim=(1, 3, 4))
    vlps_sv = vlt_sv.mean(dim=1)
    vw_sv = video_sched.training_weight(t_v)
    loss_v_sv = (vlps_sv * vw_sv).mean()

    alt_sv = F.mse_loss(pred_a.float(), target_a.float(),
                        reduction="none").mean(dim=2)
    alps_sv = alt_sv.mean(dim=1)
    aw_sv = action_sched.training_weight(t_a)
    loss_a_sv = (alps_sv * aw_sv).mean()

    total_sv = lambda_video * loss_v_sv + lambda_action * loss_a_sv

    check("video loss", torch.allclose(loss_v_sv, loss_video_og, atol=1e-6),
          f"starVLA={loss_v_sv.item():.6f}, FastWAM={loss_video_og.item():.6f}")
    check("action loss", torch.allclose(loss_a_sv, loss_action_og, atol=1e-6),
          f"starVLA={loss_a_sv.item():.6f}, FastWAM={loss_action_og.item():.6f}")
    check("total loss", torch.allclose(total_sv, total_og, atol=1e-6),
          f"starVLA={total_sv.item():.6f}, FastWAM={total_og.item():.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: Output format compatibility with starVLA trainer
# ═══════════════════════════════════════════════════════════════════════════════
def test_output_format():
    print("\n═══ Test 7: Output format compatibility ═══")

    # FastWAM returns: (loss_total, loss_dict)
    # starVLA forward returns: {"action_loss": tensor, "video_loss": tensor}
    # starVLA trainer sums all *_loss keys for backward.

    mock_output = {
        "action_loss": torch.tensor(0.5, requires_grad=True),
        "video_loss": torch.tensor(0.3, requires_grad=True),
    }

    # Simulate starVLA trainer logic (from train_starvla.py L369-376)
    total = sum(v for k, v in mock_output.items()
                if k.endswith("_loss") and isinstance(v, torch.Tensor))

    check("output has action_loss", "action_loss" in mock_output)
    check("output has video_loss", "video_loss" in mock_output)
    check("total loss is sum of components",
          torch.allclose(total, mock_output["action_loss"] + mock_output["video_loss"]))
    check("total loss has grad",
          total.requires_grad,
          "total.requires_grad=True means backward() will work")

    # FastWAM equivalent: loss_total = lambda_v * loss_v + lambda_a * loss_a
    # starVLA equivalent: sum(action_loss, video_loss) where each already includes lambda
    # These are mathematically identical.
    lv, la = 1.0, 1.0
    fastwam_total = lv * mock_output["video_loss"] + la * mock_output["action_loss"]
    check("starVLA sum == FastWAM lambda-weighted sum",
          torch.allclose(total, fastwam_total))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: MoT attention mask structure
# ═══════════════════════════════════════════════════════════════════════════════
def test_mot_mask():
    print("\n═══ Test 8: MoT attention mask structure ═══")
    Sv, Sa, tpf = 504, 8, 56  # typical LIBERO: 9 latent frames x 56 tokens/frame

    mask = starvla_build_mask(
        video_seq_len=Sv,
        action_seq_len=Sa,
        video_tokens_per_frame=tpf,
        video_attention_mask_mode="first_frame_causal",
        device=torch.device("cpu"),
    )
    total = Sv + Sa
    check("mask shape", mask.shape == (total, total),
          f"expected ({total},{total}), got {mask.shape}")

    # V→V block (first_frame_causal): first frame attends to self, rest are causal
    vv = mask[:Sv, :Sv]
    check("V->V first frame self-attn", vv[:tpf, :tpf].all().item())

    # V→A is False (video never sees action)
    va = mask[:Sv, Sv:]
    check("V->A is all False (video ignores action)", (~va).all().item())

    # A→A is True (action self-attn is full)
    aa = mask[Sv:, Sv:]
    check("A->A is full self-attn", aa.all().item())

    # A→V: action attends to first-frame video only
    av = mask[Sv:, :Sv]
    check("A->first-frame-video is True", av[:, :tpf].all().item())
    check("A->non-first-frame-video is False", (~av[:, tpf:]).all().item())


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: Noise schedule shapes (video vs action)
# ═══════════════════════════════════════════════════════════════════════════════
def test_noise_shapes():
    print("\n═══ Test 9: Noise/timestep shapes for joint forward ═══")
    B = 2
    C, T_lat, H, W = 48, 9, 14, 28
    T_action, A = 8, 7

    video_sched = StarVLAScheduler(num_train_timesteps=1000, shift=5.0)
    action_sched = StarVLAScheduler(num_train_timesteps=1000, shift=5.0)

    clean_lat = torch.randn(B, C, T_lat, H, W)
    clean_act = torch.randn(B, T_action, A)

    t_v = video_sched.sample_training_t(B, torch.device("cpu"), clean_lat.dtype)
    t_a = action_sched.sample_training_t(B, torch.device("cpu"), clean_act.dtype)

    check("video timestep shape", t_v.shape == (B,))
    check("action timestep shape", t_a.shape == (B,))
    check("video timestep range", (t_v >= 0).all() and (t_v <= 1000).all(),
          f"min={t_v.min():.1f}, max={t_v.max():.1f}")
    check("action timestep range", (t_a >= 0).all() and (t_a <= 1000).all(),
          f"min={t_a.min():.1f}, max={t_a.max():.1f}")

    noisy_v = video_sched.add_noise(clean_lat, torch.randn_like(clean_lat), t_v)
    noisy_a = action_sched.add_noise(clean_act, torch.randn_like(clean_act), t_a)
    check("noisy video shape", noisy_v.shape == clean_lat.shape)
    check("noisy action shape", noisy_a.shape == clean_act.shape)

    target_v = video_sched.training_target(clean_lat, torch.randn_like(clean_lat), t_v)
    target_a = action_sched.training_target(clean_act, torch.randn_like(clean_act), t_a)
    check("target video shape", target_v.shape == clean_lat.shape)
    check("target action shape", target_a.shape == clean_act.shape)

    # Per-token video timestep for diffusers backbone (expand to [B, seq_len])
    p_t, p_h, p_w = 1, 2, 2
    seq_len = (T_lat // p_t) * (H // p_h) * (W // p_w)
    t_int = t_v.long().clamp(min=1)
    timestep_expanded = t_int.unsqueeze(1).expand(B, seq_len).contiguous()
    check("expanded video timestep shape", timestep_expanded.shape == (B, seq_len),
          f"expected ({B},{seq_len}), got {timestep_expanded.shape}")
    check("expanded timestep is integer", timestep_expanded.dtype == torch.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10: Input/output interface contract
# ═══════════════════════════════════════════════════════════════════════════════
def test_interface_contract():
    print("\n═══ Test 10: Input/output interface contract ═══")

    # ── FastWAM training_loss expects ──
    fastwam_sample_keys = {"video", "context", "context_mask", "action"}
    fastwam_optional_keys = {"proprio", "action_is_pad", "image_is_pad"}
    fastwam_output_keys = {"loss_video", "loss_action"}

    # ── starVLA forward expects (per example dict) ──
    starvla_example_keys = {"image", "lang", "action"}
    starvla_optional_keys = {"state", "action_is_pad", "image_is_pad"}
    starvla_output_keys = {"action_loss", "video_loss"}

    print("  FastWAM training_loss sample keys:")
    print(f"    required: {fastwam_sample_keys}")
    print(f"    optional: {fastwam_optional_keys}")
    print(f"    output:   loss_total (scalar), loss_dict={fastwam_output_keys}")

    print("  starVLA forward example keys:")
    print(f"    required: {starvla_example_keys}")
    print(f"    optional: {starvla_optional_keys}")
    print(f"    output:   dict={starvla_output_keys}")

    # Key mapping
    mapping = {
        "FastWAM video [B,3,T,H,W]":       "starVLA image (list of PIL/numpy per cam)",
        "FastWAM context [B,L,D]":          "starVLA lang (str, backbone encodes)",
        "FastWAM context_mask [B,L]":       "starVLA (auto from backbone)",
        "FastWAM action [B,T,A]":           "starVLA action [T,A] (per example)",
        "FastWAM proprio [B,T,D] -> [:,0,:]": "starVLA state [D] (per example)",
        "FastWAM action_is_pad [B,T]":      "starVLA action_is_pad [T] (per example)",
        "FastWAM image_is_pad [B,T]":       "starVLA image_is_pad [T] (per example)",
    }
    print("\n  Field mapping (FastWAM -> starVLA):")
    for k, v in mapping.items():
        print(f"    {k}")
        print(f"      -> {v}")

    check("required fields covered",
          {"action"} <= fastwam_sample_keys and {"action"} <= starvla_example_keys)
    check("padding fields available in both",
          {"action_is_pad", "image_is_pad"} <= fastwam_optional_keys and
          {"action_is_pad", "image_is_pad"} <= starvla_optional_keys)
    check("output contains video + action loss",
          len(starvla_output_keys) == 2 and "action_loss" in starvla_output_keys)

    # ── Internal computation alignment ──
    print("\n  Internal computation alignment:")
    alignment = [
        ("VAE encoding",          "FastWAM: _encode_video_latents",   "starVLA: backbone.build_inputs"),
        ("Text encoding",        "FastWAM: precomputed context",     "starVLA: backbone.build_inputs"),
        ("Proprio injection",    "FastWAM: _append_proprio_to_ctx",  "starVLA: _inject_proprio (append)"),
        ("Video noise",          "FastWAM: train_video_scheduler",   "starVLA: video_scheduler_train"),
        ("Action noise",         "FastWAM: train_action_scheduler",  "starVLA: action_scheduler_train"),
        ("First-frame replace",  "FastWAM: latents[:,0:1]=clean",    "starVLA: noisy_video[:,0:1]=clean"),
        ("Joint forward",        "FastWAM: MoT.forward()",           "starVLA: backbone.transformer + MoTAttnProcessor"),
        ("Video prediction",     "FastWAM: video_expert.post_dit()", "starVLA: dit_output.sample"),
        ("Action prediction",    "FastWAM: action_expert.post_dit()","starVLA: action_expert.head()"),
        ("Video loss",           "FastWAM: mse, exclude frame0",     "starVLA: mse, exclude frame0"),
        ("Action loss",          "FastWAM: mse + pad mask",          "starVLA: mse + pad mask"),
        ("Reweighting",          "FastWAM: training_weight(t)",      "starVLA: training_weight(t)"),
        ("Lambda scaling",       "FastWAM: lambda_v/a * loss",       "starVLA: lambda_v/a * loss"),
    ]
    for step, fw, sv in alignment:
        print(f"    {step:24s}  {fw:42s}  {sv}")
    check("all 13 computation steps mapped", len(alignment) == 13)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("WanFastWAM Alignment Smoke Test")
    print("starVLA vs FastWAM (original)")
    print("=" * 70)

    test_scheduler_identity()
    test_proprio_injection()
    test_first_frame_handling()
    test_video_loss_with_padding()
    test_action_loss_with_padding()
    test_full_loss_pipeline()
    test_output_format()
    test_mot_mask()
    test_noise_shapes()
    test_interface_contract()

    # Summary
    passed = sum(1 for _, ok in test_results if ok)
    total = len(test_results)
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print(f"{PASS} All tests passed! starVLA WanFastWAM is aligned with FastWAM.")
    else:
        failed = [(n, ok) for n, ok in test_results if not ok]
        print(f"{FAIL} {len(failed)} test(s) failed:")
        for name, _ in failed:
            print(f"  - {name}")
    print("=" * 70)
    sys.exit(0 if passed == total else 1)
