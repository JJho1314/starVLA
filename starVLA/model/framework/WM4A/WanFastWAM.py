# Copyright 2025 starVLA community. All rights reserved.
"""
WanFastWAM: Wan2.2-TI2V backbone + FastWAM-aligned ActionDiT + joint MoT training.

Alignment status vs FastWAM official (LIBERO uncond):
    [done]    ActionDiT structure (30-layer DiT, hidden=1024, ~1.02B params)
    [done]    Continuous flow-matching scheduler (shift=5.0, v = noise - x_clean)
    [done]    Multi-step action denoising at inference
    [done]    Joint MoT attention (V/A concat Q/K/V via MoTAttnProcessor)
    [done]    Prefill video KV cache + action-only denoising at inference
    [done]    Proprio injection appended to text context (same as FastWAM)
    [done]    Single joint forward for both video + action losses in training
    [done]    Gaussian-shaped training_weight loss reweighting
    [done]    First-frame clean latent replacement (TI2V conditioning)
    [done]    action_is_pad / image_is_pad variable-length padding support
    [done]    Video cross-attn uses text+proprio context (same as action)
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.FastWAM_ActionDiT import ActionDiT
from starVLA.model.modules.action_model.FastWAM_Scheduler import (
    WanContinuousFlowMatchScheduler,
)
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.modules.world_model.FastWAM_MaskUtils import build_mot_attention_mask
from starVLA.model.modules.world_model.mot import MoT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class WanFastWAMDefaultConfig:
    """WanFastWAM default parameters (B4 transitional)."""

    name: str = "WanFastWAM"

    # === World Model backbone (Wan2.2-TI2V-5B) ===
    world_model: dict = field(default_factory=lambda: {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
    })

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vl_hidden_dim": 3072,
        "num_vl_layers": 30,
    })

    # === ActionDiT (FastWAM-aligned, replaces LayerwiseFM) ===
    action_dit: dict = field(default_factory=lambda: {
        "hidden_dim": 1024,
        "action_dim": 7,
        "ffn_dim": 4096,
        "text_dim": 4096,           # must match T5/UMT5-XXL output
        "freq_dim": 256,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": False,
        "pretrained_path": None,    # set to ActionDiT_linear_interp_*.pt to init from Wan22
        "skip_pretrained_load": False,
    })

    # === Action chunking + proprio ===
    action_model: dict = field(default_factory=lambda: {
        "action_dim": 7,
        "state_dim": 7,             # raw proprio dim (LIBERO=8 with gripper, set in yaml)
        "future_action_window_size": 7,
        "past_action_window_size": 0,
        "num_inference_steps": 4,
        "use_proprio": True,        # if True, project state to text_dim and prepend to context
    })

    # === Action FM scheduler (FastWAM uses shift=5.0) ===
    action_scheduler: dict = field(default_factory=lambda: {
        "num_train_timesteps": 1000,
        "train_shift": 5.0,
        "infer_shift": 5.0,
    })

    # === Video FM scheduler (aligned with FastWAM: shift=5.0) ===
    video_scheduler: dict = field(default_factory=lambda: {
        "num_train_timesteps": 1000,
        "train_shift": 5.0,
        "infer_shift": 5.0,
    })

    # === Image preprocessing (FastWAM trains 2-cam horizontally concatenated to 224x448) ===
    image_preprocess: dict = field(default_factory=lambda: {
        "image_height": 224,
        "image_width": 448,             # 2 cams × 224 horizontal concat
        "num_cameras": 2,               # if eval client passes a list of N images, concat them along width
    })

    # === Loss config (aligned with FastWAM: both lambdas default to 1.0) ===
    fastwam: dict = field(default_factory=lambda: {
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "video_loss_dtype": "bfloat16",
        "enable_video_loss": True,
        "vae_temporal_factor": 4,
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("WanFastWAM")
class Wan_FastWAM(baseframework):
    """Wan2.2 backbone + FastWAM-aligned ActionDiT + video FM aux loss (B4)."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanFastWAMDefaultConfig, config)

        # === Backbone (Wan2.2-TI2V-5B) — unchanged ===
        self.backbone = get_world_model(config=self.config)

        # === ActionDiT (replaces LayerwiseFlowmatchingActionHead) ===
        adit_cfg = dict(self.config.framework.action_dit)
        pretrained_path = adit_cfg.pop("pretrained_path", None)
        skip_pretrained = bool(adit_cfg.pop("skip_pretrained_load", False))
        if int(adit_cfg.get("text_dim", 0)) != 4096:
            logger.warning(
                f"action_dit.text_dim={adit_cfg.get('text_dim')} != 4096 (UMT5-XXL); "
                "context cross-attn input dim must match T5 encoder output."
            )
        if pretrained_path or skip_pretrained:
            self.action_expert = ActionDiT.from_pretrained(
                action_dit_config=adit_cfg,
                action_dit_pretrained_path=pretrained_path,
                skip_dit_load_from_pretrain=skip_pretrained,
                device="cpu",
                torch_dtype=torch.float32,
            )
        else:
            self.action_expert = ActionDiT(**adit_cfg)

        # === Action FM schedulers (train + infer can have different shift) ===
        sch_cfg = self.config.framework.action_scheduler
        self.action_scheduler_train = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(sch_cfg.num_train_timesteps),
            shift=float(sch_cfg.train_shift),
        )
        self.action_scheduler_infer = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(sch_cfg.num_train_timesteps),
            shift=float(sch_cfg.infer_shift),
        )

        # === Video FM schedulers (aligned with FastWAM) ===
        vsch_cfg = self.config.framework.video_scheduler
        self.video_scheduler_train = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(vsch_cfg.num_train_timesteps),
            shift=float(vsch_cfg.train_shift),
        )
        self.video_scheduler_infer = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(vsch_cfg.num_train_timesteps),
            shift=float(vsch_cfg.infer_shift),
        )

        # === Image preprocess (resolution / N cams) ===
        ip = self.config.framework.get("image_preprocess", None)
        if ip is None:
            self.image_height = 224
            self.image_width = 448
            self.num_cameras = 2
        else:
            self.image_height = int(ip.get("image_height", 224))
            self.image_width = int(ip.get("image_width", 448))
            self.num_cameras = int(ip.get("num_cameras", 2))

        # === Action chunking shape + proprio ===
        am = self.config.framework.action_model
        self.future_action_window_size = int(am.future_action_window_size)
        self.past_action_window_size = int(am.past_action_window_size)
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.action_dim = int(am.action_dim)
        self.state_dim = int(am.state_dim)
        self.num_inference_steps = int(am.num_inference_steps)
        self.use_proprio = bool(am.get("use_proprio", True))
        if self.use_proprio:
            text_dim = int(self.config.framework.action_dit.text_dim)
            # FastWAM keeps the proprio encoder in bf16 (fastwam.py:58-59 — `.to(torch_dtype)`).
            # Default fp32 here would mismatch the bf16 state we feed in via `_inject_proprio`,
            # crashing outside autocast and producing slightly different numerics inside it.
            self.proprio_encoder = nn.Linear(self.state_dim, text_dim).to(torch.bfloat16)
        else:
            self.proprio_encoder = None

        # === Loss config (aligned with FastWAM) ===
        fw = self.config.framework.get("fastwam", {})
        self.lambda_video = float(fw.get("lambda_video", 1.0))
        self.lambda_action = float(fw.get("lambda_action", 1.0))
        self.enable_video_loss = bool(fw.get("enable_video_loss", True))
        _vd = fw.get("video_loss_dtype", "bfloat16")
        self._video_dtype = torch.bfloat16 if _vd == "bfloat16" else torch.float32
        self._vae_temporal_factor = int(fw.get("vae_temporal_factor", 4))
        # MoT attention mode: "fastwam" (action sees first-frame video only; matches
        # FastWAM standard; lets inference reuse video KV cache) or "joint" (action
        # sees full video; matches FastWAMJoint; can't reuse KV cache at inference).
        self.mot_attention_mode = str(fw.get("mot_attention_mode", "fastwam"))
        if self.mot_attention_mode not in ("fastwam", "joint"):
            raise ValueError(
                f"fastwam.mot_attention_mode={self.mot_attention_mode!r}; expected 'fastwam' or 'joint'."
            )

        n_action = sum(p.numel() for p in self.action_expert.parameters()) / 1e9

        # === Install joint MoT module (FastWAM-style, replaces MoTAttnProcessor side-channel) ===
        # MoT owns the per-layer joint Q/K/V concatenation + single shared SDPA. The video
        # expert is FastWAM's bf16-clean WanVideoDiT (vendored in `world_model.wan_video_dit`),
        # the action expert is the existing ActionDiT. Both have identical num_layers, num_heads,
        # attn_head_dim — that's a hard requirement of MoT's mixed attention.
        video_blocks = self.backbone.transformer.blocks
        if len(video_blocks) != len(self.action_expert.blocks):
            raise ValueError(
                f"video blocks ({len(video_blocks)}) must equal action expert blocks "
                f"({len(self.action_expert.blocks)})"
            )
        if int(self.backbone.transformer.num_heads) != int(self.action_expert.num_heads):
            raise ValueError(
                f"num_heads must match between video DiT ({self.backbone.transformer.num_heads}) "
                f"and action DiT ({self.action_expert.num_heads}) for MoT mixed attention"
            )
        if int(self.backbone.transformer.attn_head_dim) != int(self.action_expert.attn_head_dim):
            raise ValueError(
                f"attn_head_dim must match between video DiT ({self.backbone.transformer.attn_head_dim}) "
                f"and action DiT ({self.action_expert.attn_head_dim}) for MoT mixed attention"
            )
        # MoT is a stateless orchestrator — its parameters live in the mixtures (which are
        # already registered as `self.backbone.transformer` and `self.action_expert`).
        # Bypass nn.Module's auto-registration so the parameters don't appear twice in
        # state_dict (once under their canonical path, once under `mot.mixtures.video.*`).
        _mot_instance = MoT(
            mixtures={"video": self.backbone.transformer, "action": self.action_expert},
            mot_checkpoint_mixed_attn=False,
        )
        object.__setattr__(self, "mot", _mot_instance)

        logger.info(
            f"WanFastWAM (post-C): action_expert={n_action:.2f}B, chunk_len={self.chunk_len}, "
            f"infer_steps={self.num_inference_steps}, lambda_video={self.lambda_video}, "
            f"mot_attention_mode={self.mot_attention_mode!r}, "
            f"MoT installed (FastWAM-style joint attention) on {len(video_blocks)} layers"
        )

    # ---------------------- image preprocessing ----------------------
    def _build_backbone_images(self, raw_images):
        """Convert per-example image input to a list-of-list of PIL images that
        backbone.build_inputs expects.

        `raw_images[i]` (one batch element) may be:
          1) a single image (np.ndarray HxWxC uint8 or PIL.Image)
                 → 1-frame video, single cam.
          2) a list/tuple of N camera images (single-frame, multi-cam — eval path
             where each cam is one frame)
                 → concat N cams horizontally → 1-frame video.
          3) a list/tuple of N per-cam-lists, each holding T PIL/ndarray frames
             (multi-frame training path emitted by `_pack_sample` when
             `data_cfg.multi_frame_video=true`)
                 → for each timestep t, concat cams horizontally → list of T PIL frames.

        Returns a list of length B; each element is a list of PIL Image of length T.
        """
        from PIL import Image as PILImage

        def _to_uint8_arr(x):
            if isinstance(x, PILImage.Image):
                return np.asarray(x)
            arr = np.asarray(x)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.dtype.kind == "f" else arr.astype(np.uint8)
            return arr

        def _concat_cams(cam_arrays):
            # All cams must share the same H and C; concat along width (axis=1).
            for arr in cam_arrays:
                if arr.ndim != 3:
                    raise ValueError(f"Each cam frame must be HxWxC, got shape {arr.shape}")
            Hs = {a.shape[0] for a in cam_arrays}
            Cs = {a.shape[2] for a in cam_arrays}
            if len(Hs) != 1 or len(Cs) != 1:
                raise ValueError(f"Camera shape mismatch across cams: H={Hs}, C={Cs}")
            return PILImage.fromarray(np.concatenate(cam_arrays, axis=1))

        out = []
        for img in raw_images:
            # Case 3: multi-frame, multi-cam: list of cams, each cam is a list of T frames.
            if (
                isinstance(img, (list, tuple))
                and len(img) > 0
                and isinstance(img[0], (list, tuple))
            ):
                num_cams = len(img)
                T = len(img[0])
                for ci, cam_frames in enumerate(img):
                    if len(cam_frames) != T:
                        raise ValueError(
                            f"Per-cam frame counts must match; got "
                            f"{[len(c) for c in img]} (cam {ci} differs)"
                        )
                frames_concat = []
                for t in range(T):
                    cam_arrays = [_to_uint8_arr(img[ci][t]) for ci in range(num_cams)]
                    frames_concat.append(_concat_cams(cam_arrays))
                out.append(frames_concat)
                continue

            # Case 2: single-frame, multi-cam: list of N camera ndarrays / PILs.
            if isinstance(img, (list, tuple)):
                cam_arrays = [_to_uint8_arr(one) for one in img]
                pil = _concat_cams(cam_arrays)
                out.append([pil])
                continue

            # Case 1: single image (PIL or ndarray), single cam.
            if isinstance(img, PILImage.Image):
                out.append([img])
            elif isinstance(img, np.ndarray):
                arr = _to_uint8_arr(img)
                if arr.ndim != 3:
                    raise ValueError(f"single-cam image must be HxWxC, got shape {arr.shape}")
                out.append([PILImage.fromarray(arr)])
            else:
                raise TypeError(f"unsupported image type {type(img)}")
        return out

    # ---------------------- state normalization (eval-time, optional) ----------------------
    def set_state_stats(self, stats_min, stats_max):
        """Install state min/max so predict_action can normalize raw state before
        feeding to proprio_encoder. Trainer-time state is normalized to [-1, 1]
        via min/max in FastWAM's processor; eval-time must match.
        """
        import torch as _t
        self._state_min = _t.as_tensor(stats_min, dtype=_t.float32)
        self._state_max = _t.as_tensor(stats_max, dtype=_t.float32)

    def _normalize_state(self, state_t: torch.Tensor) -> torch.Tensor:
        """Apply min/max normalization to [-1, 1] if stats are installed.

        Two callers with different conventions:
          - Trainer-eval (`train_starvla.eval_action_model`): passes batches from the
            dataloader, where state has already been normalized to [-1, 1] by the
            gr00t_lerobot processor. Stats are NOT installed here; we passthrough.
          - Deployment (`server_wanfastwam.py`): passes raw LIBERO sensor state and
            calls `set_state_stats(min, max)` at startup so we normalize internally.

        If no stats are installed we passthrough and emit a one-time warning so a
        deployment that forgot `set_state_stats()` is still surfaced (vs silently
        feeding raw state into a proprio encoder trained on normalized values).
        """
        if not hasattr(self, "_state_min") or self._state_min is None:
            if not getattr(self, "_warned_no_state_stats", False):
                import logging
                logging.warning(
                    "[WanFastWAM] _normalize_state called but no stats installed; "
                    "treating input as already-normalized. Trainer-eval path is OK; "
                    "if you are running deployment, call `model.set_state_stats(min, max)` "
                    "before inference."
                )
                self._warned_no_state_stats = True
            return state_t
        smin = self._state_min.to(device=state_t.device, dtype=state_t.dtype)
        smax = self._state_max.to(device=state_t.device, dtype=state_t.dtype)
        rng = (smax - smin).clamp(min=1e-6)
        return 2.0 * (state_t - smin) / rng - 1.0

    # ---------------------- proprio injection ----------------------
    def _inject_proprio(
        self,
        text_embeds: torch.Tensor,                # [B, L, text_dim]
        text_mask: Optional[torch.Tensor],        # [B, L] bool, or None
        state: Optional[torch.Tensor],            # [B, state_dim], or None
    ) -> tuple:
        """If proprio is enabled and state is provided, append a single
        proprio token (Linear(state_dim->text_dim)) to context (aligned with FastWAM).

        Returns (context, context_mask) ready for both video and action cross-attn.
        """
        B = text_embeds.shape[0]
        device = text_embeds.device

        if self.proprio_encoder is None or state is None or not self.use_proprio:
            if text_mask is None:
                text_mask = torch.ones(
                    (B, text_embeds.shape[1]), dtype=torch.bool, device=device
                )
            return text_embeds, text_mask

        proprio_emb = self.proprio_encoder(state.to(text_embeds.dtype)).unsqueeze(1)
        new_context = torch.cat([text_embeds, proprio_emb], dim=1)
        if text_mask is None:
            text_mask = torch.ones(
                (B, text_embeds.shape[1]), dtype=torch.bool, device=device
            )
        proprio_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        new_mask = torch.cat([text_mask, proprio_mask], dim=1)
        return new_context, new_mask

    def _compute_video_geom(self, clean_latents: torch.Tensor) -> tuple:
        """Return (video_seq_len, tokens_per_frame) given backbone input latents."""
        p_t, p_h, p_w = self.backbone.transformer.config.patch_size
        _, _, T, H, W = clean_latents.shape
        tokens_per_frame = (H // p_h) * (W // p_w)
        video_seq_len = (T // p_t) * tokens_per_frame
        return int(video_seq_len), int(tokens_per_frame)

    # ---------------------- B8: prefill + KV cache reuse ----------------------
    @torch.inference_mode()
    def _prefill_video_cache(self, wm_inputs: dict) -> tuple:
        """Run video DiT once and cache per-layer (K_video, V_video) for action denoising.

        Wraps `MoT.prefill_video_cache`: video runs alone (no joint attention with action),
        each layer returns its post-RoPE K/V which we reuse during the multi-step action
        denoising loop. The video pass is computed once per inference call; subsequent
        `_forward_action_with_kv_cache` invocations re-use these K/V caches.

        Returns:
            video_kv_cache: list of {"k": [B, Sv, H*D], "v": [B, Sv, H*D]} per layer.
            video_seq_len: int Sv
            tokens_per_frame: int
            mot_mask: full [Sv+Sa, Sv+Sa] mask shared with the action-denoise loop
        """
        clean_latents = wm_inputs["hidden_states"]
        device = clean_latents.device
        video_seq_len, tokens_per_frame = self._compute_video_geom(clean_latents)

        # Joint mask covers both prefill (sliced [:Sv,:Sv]) and action loop ([Sv:,:Sv+Sa]).
        mot_mask = build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=self.chunk_len,
            video_tokens_per_frame=tokens_per_frame,
            video_attention_mask_mode="first_frame_causal",
            mot_attention_mode=self.mot_attention_mode,
            device=device,
        )

        # Video pre_dit at timestep=0 (clean prefix: TI2V conditioning relies on the latent
        # already being clean for the full prefill run).
        timestep_video = wm_inputs.get("timestep", None)
        if timestep_video is None:
            timestep_video = torch.zeros((clean_latents.shape[0],), dtype=clean_latents.dtype, device=device)
        else:
            # build_inputs returned per-token zeros [B, Sv]; pre_dit expects [B] for fuse-vae.
            timestep_video = torch.zeros((clean_latents.shape[0],), dtype=clean_latents.dtype, device=device)

        video_pre = self.backbone.transformer.pre_dit(
            x=clean_latents,
            timestep=timestep_video,
            context=wm_inputs["encoder_hidden_states"],
            context_mask=wm_inputs.get("encoder_attention_mask", None),
            fuse_vae_embedding_in_latents=True,
        )

        # Slice mask to video-only [Sv, Sv] for the prefill self-attention.
        video_self_mask = mot_mask[:video_seq_len, :video_seq_len]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=video_self_mask,
            )
        return video_kv_cache, video_seq_len, tokens_per_frame, mot_mask

    def _forward_action_with_kv_cache(
        self,
        action_tokens_init: torch.Tensor,           # [B, chunk_len, action_dim] noisy action latent
        timestep_action: torch.Tensor,              # [B] FM timestep
        text_context_with_proprio: torch.Tensor,    # [B, L, 4096]
        text_mask_with_proprio: torch.Tensor,       # [B, L]
        video_kv_cache: list,                       # from _prefill_video_cache
        mot_mask: torch.Tensor,                     # [Sv+Sa, Sv+Sa]
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action expert through all layers using cached video K/V.

        Wraps `MoT.forward_action_with_video_cache`: at each layer the action expert
        builds its own Q/K/V, concatenates with the layer's cached video K/V, runs SDPA
        over the joint sequence with `mot_mask`, then runs action post-attention
        (cross-attn(text+proprio) + FFN). Video does NOT run again — the full inference
        cost is 1 prefill + N action-denoise loops.
        """
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_tokens_init,
            timestep=timestep_action,
            context=text_context_with_proprio,
            context_mask=text_mask_with_proprio,
        )
        return self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=mot_mask,
            video_seq_len=video_seq_len,
        )

    # ---------------------- joint MoT training loss (aligned with FastWAM) ----------------------
    def _joint_training_loss(
        self,
        actions_target: torch.Tensor,               # [B, chunk_len, action_dim]
        clean_latents: torch.Tensor,                 # [B, C, T, H, W]
        ctx: torch.Tensor,                           # [B, L, 4096] text+proprio
        ctx_mask: Optional[torch.Tensor] = None,     # [B, L] bool
        action_is_pad: Optional[torch.Tensor] = None,  # [B, chunk_len] bool
        image_is_pad: Optional[torch.Tensor] = None,   # [B, num_frames] bool
    ) -> dict:
        """Joint video+action flow-matching loss through one MoT backbone forward.

        Aligned with FastWAM's `training_loss`:
          - Video and action are noised independently with shift-based sampling.
          - First-frame latent is replaced with clean VAE latent (TI2V conditioning).
          - Both run through a single MoT forward (MoTAttnProcessor handles joint attn).
          - Video and action losses use timestep-dependent Gaussian reweighting.
          - Padding masks (action_is_pad, image_is_pad) are supported.
        """
        B = actions_target.shape[0]
        device = actions_target.device
        _, _, T, H, W = clean_latents.shape

        # === Video noise (shift-based sampling via video_scheduler) ===
        noise_video = torch.randn_like(clean_latents)
        t_video = self.video_scheduler_train.sample_training_t(B, device, clean_latents.dtype)
        noisy_video = self.video_scheduler_train.add_noise(clean_latents, noise_video, t_video)
        target_video = self.video_scheduler_train.training_target(clean_latents, noise_video, t_video)

        # First-frame: replace with clean latent (TI2V conditioning, same as FastWAM)
        first_frame_latents = clean_latents[:, :, 0:1].clone()
        noisy_video[:, :, 0:1] = first_frame_latents

        # === Action noise ===
        noise_action = torch.randn_like(actions_target)
        t_action = self.action_scheduler_train.sample_training_t(B, device, actions_target.dtype)
        noisy_action = self.action_scheduler_train.add_noise(actions_target, noise_action, t_action)
        target_action = self.action_scheduler_train.training_target(actions_target, noise_action, t_action)

        if ctx_mask is None:
            ctx_mask = torch.ones((B, ctx.shape[1]), dtype=torch.bool, device=device)

        video_seq_len, tokens_per_frame = self._compute_video_geom(clean_latents)
        Sa = noisy_action.shape[1]

        # === Per-expert pre_dit ===
        # WanVideoDiT.pre_dit handles per-token timestep expansion + first-frame timestep=0
        # internally via fuse_vae_embedding_in_latents=True (matches Wan2.2 TI2V mode).
        # NOTE: pass timestep as float (not long) — sinusoidal_embedding_1d propagates
        # `position.dtype` to its output, and the time_embedding Linear is bf16. Casting
        # to long here would yield Long input to bf16 Linear → dtype mismatch error.
        video_pre = self.backbone.transformer.pre_dit(
            x=noisy_video,
            timestep=t_video,
            context=ctx,
            context_mask=ctx_mask,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=t_action,
            context=ctx,
            context_mask=ctx_mask,
        )

        # Joint MoT mask (Sv prefix attends to its own + first-frame causal pattern; action
        # tokens get a tail row block).
        mot_mask = build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=Sa,
            video_tokens_per_frame=tokens_per_frame,
            video_attention_mask_mode="first_frame_causal",
            mot_attention_mode=self.mot_attention_mode,
            device=device,
        )

        # === Joint MoT forward (single shared SDPA per layer) ===
        with torch.autocast("cuda", dtype=self._video_dtype):
            mot_out = self.mot.forward(
                embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
                attention_mask=mot_mask,
                freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                context_all={
                    "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                    "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                },
                t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            )
            v_pred_video = self.backbone.transformer.post_dit(mot_out["video"], video_pre)
            pred_action = self.action_expert.head(mot_out["action"])

        # === Video loss (exclude first frame, padding mask, reweighting) ===
        # Single-observation training (T_latent==1) leaves no future frames after dropping
        # the clean first frame → empty tensor → MSE returns NaN (0/0). Guard explicitly so
        # video_loss reads as a clean 0 instead of NaN polluting metrics. Action loss is
        # unaffected — joint MoT still propagates video features into the action expert via
        # shared self-attention.
        if v_pred_video.shape[2] <= 1:
            # Single-observation training: nothing to supervise. Cosmetic 0.
            loss_video = torch.zeros((), device=device, dtype=torch.float32)
        else:
            pred_vid = v_pred_video[:, :, 1:]
            target_vid = target_video[:, :, 1:]
            video_loss_token = F.mse_loss(
                pred_vid.float(), target_vid.float(), reduction="none"
            ).mean(dim=(1, 3, 4))  # [B, T_latent-1]

            if image_is_pad is not None:
                tf = self._vae_temporal_factor
                tail_is_pad = image_is_pad[:, 1:]
                if tail_is_pad.shape[1] % tf == 0:
                    latent_tail_is_pad = tail_is_pad.view(B, -1, tf).all(dim=2)
                else:
                    latent_tail_is_pad = tail_is_pad[:, :video_loss_token.shape[1]]
                if latent_tail_is_pad.shape[1] == video_loss_token.shape[1]:
                    valid_v = (~latent_tail_is_pad).to(dtype=video_loss_token.dtype, device=device)
                    valid_v_sum = valid_v.sum(dim=1).clamp(min=1.0)
                    video_loss_per_sample = (video_loss_token * valid_v).sum(dim=1) / valid_v_sum
                else:
                    video_loss_per_sample = video_loss_token.mean(dim=1)
            else:
                video_loss_per_sample = video_loss_token.mean(dim=1)

            video_weight = self.video_scheduler_train.training_weight(t_video).to(
                video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
            )
            loss_video = (video_loss_per_sample * video_weight).mean()

        # === Action loss (padding mask, reweighting) ===
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)  # [B, T_action]

        if action_is_pad is not None:
            valid_a = (~action_is_pad).to(dtype=action_loss_token.dtype, device=device)
            valid_a_sum = valid_a.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid_a).sum(dim=1) / valid_a_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.action_scheduler_train.training_weight(t_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        return {
            "action_loss": self.lambda_action * loss_action,
            "video_loss": self.lambda_video * loss_video,
        }

    # ---------------------- training forward ----------------------
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        raw_images = [example["image"] for example in examples]
        batch_images = self._build_backbone_images(raw_images)
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )

        wm_inputs = self.backbone.build_inputs(
            images=batch_images,
            instructions=instructions,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        clean_latents = wm_inputs["hidden_states"]   # [B, C, T, H', W']
        text_embeds = wm_inputs["encoder_hidden_states"]  # [B, L, 4096]
        text_mask = wm_inputs.get("encoder_attention_mask", None)
        device = clean_latents.device

        actions_t = torch.tensor(np.array(actions), device=device, dtype=torch.float32)
        actions_target = actions_t[:, -self.chunk_len:, :]   # [B, chunk_len, action_dim]

        # Inject proprio into context (appended, aligned with FastWAM)
        state_t = None
        if state is not None:
            state_t = torch.tensor(np.array(state), device=device, dtype=torch.float32)
            if state_t.dim() == 3:
                state_t = state_t.squeeze(1)
        ctx, ctx_mask = self._inject_proprio(text_embeds, text_mask, state_t)

        # Optional padding masks from dataset
        action_is_pad = None
        if "action_is_pad" in examples[0]:
            action_is_pad = torch.tensor(
                np.array([ex["action_is_pad"] for ex in examples]),
                device=device, dtype=torch.bool,
            )
            action_is_pad = action_is_pad[:, -self.chunk_len:]

        image_is_pad = None
        if "image_is_pad" in examples[0]:
            image_is_pad = torch.tensor(
                np.array([ex["image_is_pad"] for ex in examples]),
                device=device, dtype=torch.bool,
            )

        # === Joint MoT training (single backbone forward for both video + action) ===
        return self._joint_training_loss(
            actions_target=actions_target,
            clean_latents=clean_latents,
            ctx=ctx,
            ctx_mask=ctx_mask,
            action_is_pad=action_is_pad,
            image_is_pad=image_is_pad,
        )

    # ---------------------- inference ----------------------
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        # Concat N cams horizontally if eval client passed list-of-cams; else use as-is.
        # Also wraps each into a 1-frame video for backbone.
        raw_images = [example["image"] for example in examples]
        batch_images = self._build_backbone_images(raw_images)
        instructions = [example["lang"] for example in examples]

        wm_inputs = self.backbone.build_inputs(
            images=batch_images,
            instructions=instructions,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        text_embeds = wm_inputs["encoder_hidden_states"]
        text_mask = wm_inputs.get("encoder_attention_mask", None)
        B = len(examples)

        # state from examples (LIBERO eval-time provides it under "state" or "observation.state")
        state_raw = []
        for ex in examples:
            s = ex.get("state", None)
            if s is None:
                s = ex.get("observation.state", None)
            state_raw.append(s)
        state_raw = [s for s in state_raw if s is not None]
        state_t = (
            torch.tensor(np.array(state_raw), device=text_embeds.device, dtype=torch.float32)
            if len(state_raw) == B
            else None
        )
        if state_t is not None and state_t.dim() == 3:
            state_t = state_t.squeeze(1)
        # Normalize state via min/max (matches FastWAM training-time processor).
        if state_t is not None:
            state_t = self._normalize_state(state_t)
        text_embeds, text_mask = self._inject_proprio(text_embeds, text_mask, state_t)
        # Video prefill must condition on the SAME text+proprio context as training:
        # _joint_training_loss feeds the proprio-injected `ctx` to the video pre_dit,
        # so the action expert attends to video K/V built with proprio. Without writing
        # the injected context back here, _prefill_video_cache would use text-only
        # context (wm_inputs["encoder_hidden_states"]), shifting the video tokens and
        # collapsing closed-loop action quality.
        wm_inputs["encoder_hidden_states"] = text_embeds
        wm_inputs["encoder_attention_mask"] = text_mask
        device = text_embeds.device

        # Initialize action latents from N(0, 1).
        # FastWAM upstream `infer_action` uses rand_device="cpu", seed=None
        # → CPU default RNG, varying noise per trial. We match that (drop the
        # over-correction of manual_seed(42) which had fixed noise across trials
        # and hurt libero_10 by -1.4pt).
        latents_action = torch.randn(
            (B, self.chunk_len, self.action_dim),
            device="cpu", dtype=torch.float32,
        ).to(device=device, dtype=text_embeds.dtype)

        # === B8: prefill video KV cache once (1 backbone forward) ===
        video_kv_cache, video_seq_len, tokens_per_frame, mot_mask = self._prefill_video_cache(wm_inputs)

        # === Multi-step flow-matching denoise; each step is action-only ===
        ts, deltas = self.action_scheduler_infer.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=device,
            dtype=latents_action.dtype,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for step_t, step_dt in zip(ts, deltas):
                t_in = step_t.unsqueeze(0).expand(B).contiguous()
                action_hidden = self._forward_action_with_kv_cache(
                    action_tokens_init=latents_action,
                    timestep_action=t_in,
                    text_context_with_proprio=text_embeds,
                    text_mask_with_proprio=text_mask,
                    video_kv_cache=video_kv_cache,
                    mot_mask=mot_mask,
                    video_seq_len=video_seq_len,
                )
                v_pred = self.action_expert.head(action_hidden)
                latents_action = self.action_scheduler_infer.step(v_pred, step_dt, latents_action)

        return {"normalized_actions": latents_action.detach().cpu().float().numpy()}

    # ============================================================================
    # FastWAMJoint-faithful inference (separate from the standard `predict_action`
    # path; only invoke for ckpts trained with `mot_attention_mode: joint`).
    # ============================================================================

    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,     # [B, z_dim, T_lat, H_lat, W_lat]
        latents_action: torch.Tensor,    # [B, chunk_len, action_dim]
        timestep_video: torch.Tensor,    # [B]
        timestep_action: torch.Tensor,   # [B]
        context: torch.Tensor,            # [B, L, text_dim] (already proprio-injected)
        context_mask: torch.Tensor,       # [B, L] bool
    ) -> tuple:
        """One joint MoT forward (no KV cache reuse).

        Mirrors upstream `FastWAMJoint._predict_joint_noise`. Returns
        (v_pred_video, v_pred_action) — flow-matching velocity predictions for
        both modalities, ready to be fed into the respective schedulers' `step`.
        """
        video_seq_len, tokens_per_frame = self._compute_video_geom(latents_video)
        Sa = latents_action.shape[1]
        device = latents_video.device

        video_pre = self.backbone.transformer.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        mot_mask = build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=Sa,
            video_tokens_per_frame=tokens_per_frame,
            video_attention_mask_mode="first_frame_causal",
            mot_attention_mode=self.mot_attention_mode,
            device=device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mot_out = self.mot.forward(
                embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
                attention_mask=mot_mask,
                freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                context_all={
                    "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                    "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                },
                t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            )
            v_pred_video = self.backbone.transformer.post_dit(mot_out["video"], video_pre)
            v_pred_action = self.action_expert.head(mot_out["action"])
        return v_pred_video, v_pred_action

    @torch.inference_mode()
    def predict_action_joint(self, examples: List[dict], **kwargs) -> dict:
        """FastWAMJoint-faithful inference: per-step joint denoise of video+action.

        Mirrors `FastWAMJoint.infer_action` upstream
        (`fastwam/models/wan22/fastwam_joint.py:96`):
          - Initialize BOTH `latents_video` and `latents_action` from N(0,1)
            on CPU (unseeded, matches FastWAM upstream rand_device="cpu", seed=None).
          - Replace `latents_video[:,:,0:1]` with the encoded clean first-frame
            latent (TI2V conditioning).
          - At each of `num_inference_steps` steps: run ONE joint MoT forward
            (no KV cache reuse), step both schedulers, re-pin the first frame.
          - Return the final action chunk.

        Cost: O(num_inference_steps × full MoT forward) — strictly worse than the
        prefill+cache fast path (`predict_action`). Only use this when the ckpt
        was trained with `mot_attention_mode: joint` AND you specifically want to
        match upstream FastWAMJoint inference numerically.

        Required config (read from `framework.fastwam`):
          - `num_video_frames` (default 9): number of pixel-space video frames
            used during training. Determines temporal latent length via
            `latent_t = (num_video_frames - 1) // vae_temporal_factor + 1`.
        """
        import logging as _logging
        if self.mot_attention_mode != "joint":
            _logging.warning(
                "[predict_action_joint] mot_attention_mode=%r (expected 'joint'). "
                "Running with full A→V mask anyway; ckpt was likely NOT trained for this "
                "and SR will degrade.",
                self.mot_attention_mode,
            )

        if type(examples) is not list:
            examples = [examples]
        raw_images = [example["image"] for example in examples]
        batch_images = self._build_backbone_images(raw_images)
        instructions = [example["lang"] for example in examples]

        wm_inputs = self.backbone.build_inputs(
            images=batch_images,
            instructions=instructions,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        text_embeds = wm_inputs["encoder_hidden_states"]
        text_mask = wm_inputs.get("encoder_attention_mask", None)
        B = len(examples)
        device = text_embeds.device
        dtype = text_embeds.dtype

        # State / proprio (same as predict_action).
        state_raw = []
        for ex in examples:
            s = ex.get("state", None)
            if s is None:
                s = ex.get("observation.state", None)
            state_raw.append(s)
        state_raw = [s for s in state_raw if s is not None]
        state_t = (
            torch.tensor(np.array(state_raw), device=device, dtype=torch.float32)
            if len(state_raw) == B
            else None
        )
        if state_t is not None and state_t.dim() == 3:
            state_t = state_t.squeeze(1)
        if state_t is not None:
            state_t = self._normalize_state(state_t)
        text_embeds, text_mask = self._inject_proprio(text_embeds, text_mask, state_t)

        # First-frame VAE latent (already encoded by build_inputs; shape [B, z_dim, 1, H_lat, W_lat]).
        first_frame_latents = wm_inputs["hidden_states"]
        if first_frame_latents.shape[2] != 1:
            # build_inputs is called with single-frame images, so temporal dim should be 1.
            # If a future config feeds multi-frame, we still only treat frame 0 as the clean anchor.
            first_frame_latents = first_frame_latents[:, :, 0:1].contiguous()
        _, z_dim, _, H_lat, W_lat = first_frame_latents.shape

        # Temporal latent length, derived from num_video_frames + vae_temporal_factor.
        fw_cfg = self.config.framework.get("fastwam", {})
        num_video_frames = int(fw_cfg.get("num_video_frames", 9))
        if (num_video_frames - 1) % self._vae_temporal_factor != 0:
            raise ValueError(
                f"num_video_frames-1 ({num_video_frames - 1}) must be divisible by "
                f"vae_temporal_factor ({self._vae_temporal_factor}) for VAE-aligned latent_t."
            )
        latent_t = (num_video_frames - 1) // self._vae_temporal_factor + 1
        if latent_t < 1:
            raise ValueError(f"latent_t derived as {latent_t} from num_video_frames={num_video_frames}")

        # CPU unseeded RNG matches FastWAM upstream `infer_action(seed=None, rand_device='cpu')`.
        latents_video = torch.randn(
            (B, z_dim, latent_t, H_lat, W_lat),
            device="cpu", dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        latents_action = torch.randn(
            (B, self.chunk_len, self.action_dim),
            device="cpu", dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        # Pin first frame to clean latent (TI2V anchor).
        latents_video[:, :, 0:1] = first_frame_latents.to(device=device, dtype=dtype)

        ts_v, deltas_v = self.video_scheduler_infer.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=device, dtype=latents_video.dtype,
        )
        ts_a, deltas_a = self.action_scheduler_infer.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=device, dtype=latents_action.dtype,
        )
        for step_t_v, step_dt_v, step_t_a, step_dt_a in zip(ts_v, deltas_v, ts_a, deltas_a):
            t_in_v = step_t_v.unsqueeze(0).expand(B).contiguous().to(latents_video.dtype)
            t_in_a = step_t_a.unsqueeze(0).expand(B).contiguous().to(latents_action.dtype)
            v_pred_video, v_pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=t_in_v,
                timestep_action=t_in_a,
                context=text_embeds,
                context_mask=text_mask,
            )
            latents_video = self.video_scheduler_infer.step(v_pred_video, step_dt_v, latents_video)
            latents_action = self.action_scheduler_infer.step(v_pred_action, step_dt_a, latents_action)
            # Re-pin first frame (it never gets denoised; matches upstream).
            latents_video[:, :, 0:1] = first_frame_latents.to(device=device, dtype=latents_video.dtype)

        return {"normalized_actions": latents_action.detach().cpu().float().numpy()}
