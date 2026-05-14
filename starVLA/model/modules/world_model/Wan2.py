# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Wan2.2-TI2V World Model Interface (post-C-refactor).

Holds:
  - tokenizer + text_encoder (UMT5-XXL, frozen) — diffusers
  - vae (AutoencoderKLWan, frozen) — diffusers
  - scheduler — diffusers
  - **transformer** = vendored FastWAM `WanVideoDiT` (replaces diffusers'
    `WanTransformer3DModel`). The diffusers implementation casts to fp32 5-6× per
    block in WanTransformerBlock.forward (norm1/norm2/norm3 + residual adds + ff
    output), inflating activation memory by ~14 GiB at bs=16 across 30 blocks.
    FastWAM's WanVideoDiT stays in bf16 throughout — same math, much less memory,
    matches what the FastWAM training/eval pipeline actually uses.

Weights are loaded from a Wan2.2-TI2V-5B-Diffusers folder at startup and remapped
(diffusers key names → FastWAM key names) using `verify_alignment_keymap.pt`.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

from .wan_video_dit import WanVideoDiT

logger = initialize_overwatch(__name__)
_keymap_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults for Wan2.2-TI2V-5B (matches FastWAM `configs/model/fastwam.yaml`).
# ---------------------------------------------------------------------------
_WAN22_TI2V_5B_DEFAULT = {
    "hidden_dim": 3072,
    "in_dim": 48,
    "ffn_dim": 14336,
    "out_dim": 48,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1.0e-6,
    "patch_size": (1, 2, 2),
    "num_heads": 24,
    "attn_head_dim": 128,
    "num_layers": 30,
    "has_image_input": False,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
    "video_attention_mask_mode": "first_frame_causal",
    "action_conditioned": False,
    "use_gradient_checkpointing": False,
}


class _TransformerConfigShim:
    """Compat shim: `transformer.config.patch_size` etc. still readable for
    callers that haven't been refactored to read attributes off the WanVideoDiT
    instance directly."""

    def __init__(self, cfg: dict):
        self.patch_size = tuple(cfg.get("patch_size", (1, 2, 2)))
        self.num_attention_heads = int(cfg.get("num_heads"))
        self.attention_head_dim = int(cfg.get("attn_head_dim"))
        self.in_channels = int(cfg.get("in_dim"))
        self.out_channels = int(cfg.get("out_dim"))


class _Wan2_Interface(nn.Module):
    """World-model wrapper for Wan2.2-TI2V-5B.

    Public API consumed by `WanFastWAM`:
      - `self.transformer`: `WanVideoDiT` instance (FastWAM-flavour, bf16-clean)
      - `self.vae`, `self.text_encoder`, `self.tokenizer`, `self.scheduler`,
        `self.video_processor`: diffusers components, frozen by default
      - `build_inputs(images, instructions, image_height, image_width)`:
        encodes inputs → dict with `hidden_states` (VAE latents),
        `encoder_hidden_states` (text embeds), `timestep` (per-token zeros).
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        )
        self.config = config

        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.video_processor import VideoProcessor
        from transformers import T5TokenizerFast, UMT5EncoderModel

        logger.info(f"Loading Wan2.2-TI2V (frozen IO modules) from {model_name}")

        # Tokenizer + text_encoder (frozen).
        # When `text_embed_cache_path` is set in the world_model config, we load a
        # pre-computed `{task_string: {"embed", "mask"}}` dict instead of UMT5-XXL.
        # UMT5-XXL is ~11 GB in bf16 — skipping it on every GPU enables larger
        # per-device batch sizes (FastWAM does the same via `load_text_encoder: false`).
        cache_path = wm_cfg.get("text_embed_cache_path", None)
        self._text_embed_cache: Optional[dict] = None
        self._text_embed_cache_dir: Optional[str] = None
        self._text_prompt_template = wm_cfg.get("text_prompt_template", None)
        self._zero_pad_text_embeds = bool(wm_cfg.get("zero_pad_text_embeds", False))
        self._force_text_mask_ones = bool(wm_cfg.get("force_text_mask_ones", False))
        if cache_path:
            cache_path = str(cache_path)
            from pathlib import Path as _Path
            cache_obj = _Path(cache_path)
            if not cache_obj.exists():
                raise FileNotFoundError(
                    f"text_embed_cache_path={cache_path} does not exist. "
                    f"Run `scripts/precompute_libero_text_embeds.py` first."
                )
            if cache_obj.is_dir():
                logger.info(f"Using FastWAM-style text embed cache dir {cache_path} (skipping UMT5)")
                self._text_embed_cache_dir = cache_path
                self._text_embed_max_length = int(wm_cfg.get("text_embed_max_length", 128))
            else:
                logger.info(f"Loading pre-computed text embeds from {cache_path} (skipping UMT5)")
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                self._text_embed_cache = payload["cache"]
                self._text_embed_max_length = int(payload.get("max_length", 128))
            self.tokenizer = None
            self.text_encoder = None
        else:
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer")
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
            )

        # VAE (frozen, used for latent encoding/decoding only)
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # Scheduler (kept around for diffusers-pipeline compat / generate())
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # Image preprocess (PIL → tensor in [-1, 1])
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # === Transformer: FastWAM's bf16-clean WanVideoDiT ===
        video_dit_cfg = self._build_video_dit_config(wm_cfg)
        logger.info(
            f"Instantiating WanVideoDiT (FastWAM-flavour) layers={video_dit_cfg['num_layers']} "
            f"hidden={video_dit_cfg['hidden_dim']} grad_ckpt={video_dit_cfg['use_gradient_checkpointing']}"
        )
        self.transformer = WanVideoDiT(**video_dit_cfg).to(torch.bfloat16)

        if not bool(wm_cfg.get("skip_transformer_load", False)):
            self._load_transformer_from_diffusers(model_name)
        else:
            logger.warning("skip_transformer_load=True; transformer weights are randomly initialized")

        # Compat: callers may still read `transformer.config.patch_size`
        self.transformer.config = _TransformerConfigShim(video_dit_cfg)

        self._hidden_size = int(video_dit_cfg["num_heads"]) * int(video_dit_cfg["attn_head_dim"])

        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_video_dit_config(wm_cfg) -> dict:
        cfg = dict(_WAN22_TI2V_5B_DEFAULT)
        user = wm_cfg.get("video_dit_config", None)
        if user:
            try:
                from omegaconf import OmegaConf
                user = OmegaConf.to_container(user, resolve=True) if not isinstance(user, dict) else user
            except Exception:
                pass
            if isinstance(user, dict):
                cfg.update(user)
        cfg["patch_size"] = tuple(cfg["patch_size"])
        return cfg

    # -----------------------------------------------------------------------
    # Weight loader: diffusers safetensors → WanVideoDiT keys
    # -----------------------------------------------------------------------
    def _load_transformer_from_diffusers(self, model_name: str) -> None:
        from safetensors.torch import load_file

        # Locate keymap (sits at repo root next to verify_alignment.py)
        here = os.path.dirname(os.path.abspath(__file__))
        # walk up: world_model → modules → model → starVLA → repo_root
        repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        keymap_path = os.path.join(repo_root, "verify_alignment_keymap.pt")
        if not os.path.exists(keymap_path):
            raise FileNotFoundError(
                f"verify_alignment_keymap.pt not found at {keymap_path}; "
                f"required to convert diffusers weights → WanVideoDiT format."
            )
        km = torch.load(keymap_path, weights_only=False)
        # km["video"] = {fastwam_key (mot prefix): diffusers_key}
        # WanVideoDiT keys = fastwam_key without "mixtures.video." prefix
        diff2wv = {}
        for fk, dk in km["video"].items():
            if fk.startswith("mixtures.video."):
                wv_key = fk[len("mixtures.video."):]
            else:
                wv_key = fk
            diff2wv[dk] = wv_key

        # Locate transformer safetensors shards
        tx_dir = os.path.join(model_name, "transformer")
        files = sorted(glob.glob(os.path.join(tx_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(
                f"No transformer/*.safetensors under {tx_dir}; cannot load WanVideoDiT."
            )

        diffusers_state: dict[str, torch.Tensor] = {}
        for f in files:
            diffusers_state.update(load_file(f))

        wv_state: dict[str, torch.Tensor] = {}
        unmapped: list[str] = []
        for dk, tensor in diffusers_state.items():
            wv_key = diff2wv.get(dk)
            if wv_key is None:
                unmapped.append(dk)
                continue
            wv_state[wv_key] = tensor.to(torch.bfloat16)

        # Strict load against the freshly-built WanVideoDiT
        missing, unexpected = self.transformer.load_state_dict(wv_state, strict=False)
        if missing or unexpected or unmapped:
            logger.warning(
                f"WanVideoDiT load: matched={len(wv_state)} missing={len(missing)} "
                f"unexpected={len(unexpected)} unmapped_diffusers_keys={len(unmapped)}"
            )
            for k in missing[:5]:
                logger.warning(f"  [missing] {k}")
            for k in unexpected[:5]:
                logger.warning(f"  [unexpected] {k}")
            for k in unmapped[:5]:
                logger.warning(f"  [unmapped diffusers key] {k}")
        else:
            logger.info(f"WanVideoDiT: strict-loaded {len(wv_state)} tensors from {tx_dir} ✓")

    # -----------------------------------------------------------------------
    # Compat property used by base framework
    # -----------------------------------------------------------------------
    @property
    def model(self):
        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    # -----------------------------------------------------------------------
    # Encoders
    # -----------------------------------------------------------------------
    def _format_text_instruction(self, instruction: str) -> str:
        text = str(instruction).strip()
        template = self._text_prompt_template
        if template:
            return str(template).format(task=text)
        return text

    def _fastwam_align_text_context(self, text_embeds: torch.Tensor, text_mask: torch.Tensor):
        if self._zero_pad_text_embeds:
            text_embeds = text_embeds.clone()
            text_embeds[~text_mask] = 0.0
        if self._force_text_mask_ones:
            text_mask = torch.ones_like(text_mask, dtype=torch.bool)
        return text_embeds, text_mask

    def _load_text_context_from_dir(self, prompt: str, max_length: int):
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            self._text_embed_cache_dir,
            f"{hashed}.t5_len{max_length}.wan22ti2v5b.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing FastWAM text embedding cache: {cache_path}. "
                f"Prompt was: {prompt!r}"
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["context"], payload["mask"]

    def _encode_text(self, instructions, max_length=128):
        """Encode prompts; returns (embeds [B,L,D] bf16, mask [B,L] bool).

        Two paths:
          - Cached (preferred): `text_embed_cache_path` was set at init → look up
            (embed, mask) from precomputed dict, no UMT5 on GPU. Saves ~11 GB.
          - Online: encode via UMT5 (bf16). Used when no cache provided.

        ``max_length=128`` matches FastWAM's `context_len: 128`.
        """
        formatted_instructions = [self._format_text_instruction(inst) for inst in instructions]

        if self._text_embed_cache is not None or self._text_embed_cache_dir is not None:
            if max_length != self._text_embed_max_length:
                raise ValueError(
                    f"Cached text embeds have max_length={self._text_embed_max_length} "
                    f"but caller asked for {max_length}. Re-run precompute with the "
                    f"new max_length or fix the call site."
                )
            # Use VAE's device since text encoder is gone.
            device = next(self.vae.parameters()).device
            embeds_list, masks_list = [], []
            for key in formatted_instructions:
                if self._text_embed_cache_dir is not None:
                    embed, mask = self._load_text_context_from_dir(key, max_length)
                else:
                    if key not in self._text_embed_cache:
                        raise KeyError(
                            f"Instruction not found in text-embed cache: {key!r}. "
                            f"Re-run `scripts/precompute_libero_text_embeds.py` to include "
                            f"this string, or unset `text_embed_cache_path` to fall back to UMT5."
                        )
                    entry = self._text_embed_cache[key]
                    embed, mask = entry["embed"], entry["mask"]
                embeds_list.append(embed)  # [L, D]
                masks_list.append(mask)    # [L]
            text_embeds = torch.stack(embeds_list, dim=0).to(device=device, dtype=torch.bfloat16)
            text_mask = torch.stack(masks_list, dim=0).to(device=device, dtype=torch.bool)
            return self._fastwam_align_text_context(text_embeds, text_mask)

        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            formatted_instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state
        text_mask = text_inputs.attention_mask.to(dtype=torch.bool)
        text_embeds = text_embeds.to(dtype=torch.bfloat16)
        return self._fastwam_align_text_context(text_embeds, text_mask)

    def _encode_images_vae(self, images, num_frames=None, image_height=None, image_width=None):
        """Encode observation images through VAE to latents [B, 48, T_lat, H/16, W/16]."""
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        height = int(image_height) if image_height is not None else 480
        width = int(image_width) if image_width is not None else 832

        preprocessed, frame_counts = [], []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]
            video_tensor = self.video_processor.preprocess_video(
                sample_imgs, height=height, width=width
            ).to(device=device, dtype=dtype)
            preprocessed.append(video_tensor)
            frame_counts.append(video_tensor.shape[2])

        target_frames = num_frames if num_frames is not None else max(frame_counts)
        batch_videos = []
        for v in preprocessed:
            n = v.shape[2]
            if n > target_frames:
                v = v[:, :, :target_frames]
            elif n < target_frames:
                last = v[:, :, -1:]
                pad = last.repeat(1, 1, target_frames - n, 1, 1)
                v = torch.cat([v, pad], dim=2)
            batch_videos.append(v.squeeze(0))

        video = torch.stack(batch_videos, dim=0)
        with torch.no_grad():
            latents = self.vae.encode(video).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) * latents_std
        return latents

    def build_inputs(self, images, instructions, image_height=None, image_width=None, **kwargs):
        """Encode (images, instructions) → dict consumed by `WanFastWAM` joint training."""
        assert len(images) == len(instructions)
        text_embeds, text_mask = self._encode_text(instructions)
        latents = self._encode_images_vae(images, image_height=image_height, image_width=image_width)

        batch_size = latents.shape[0]
        device = latents.device
        p_t, p_h, p_w = self.transformer.config.patch_size
        _, _, T, H, W = latents.shape
        seq_len = (T // p_t) * (H // p_h) * (W // p_w)
        timestep = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "encoder_attention_mask": text_mask,
            "_is_wm_input": True,
        }

    # -----------------------------------------------------------------------
    # forward() is intentionally NOT implemented here.
    # WanVideoDiT is consumed via `pre_dit`/MoT/`post_dit` rather than a single
    # monolithic forward call. Callers that need standalone DiT forward should
    # use `WanVideoDiT.pre_dit + post_dit + per-block` directly.
    # -----------------------------------------------------------------------
    def forward(self, **kwargs):
        raise NotImplementedError(
            "Wan2.transformer is FastWAM's WanVideoDiT, which does not expose a monolithic "
            "forward(). Use the MoT-driven joint flow in WanFastWAM, or call pre_dit + "
            "post_dit explicitly."
        )

    # -----------------------------------------------------------------------
    # generate() — diffusers-pipeline shortcut, retained for compat (no-op for training)
    # -----------------------------------------------------------------------
    def generate(self, **kwargs):
        raise NotImplementedError(
            "Standalone WanPipeline.generate() requires diffusers WanTransformer3DModel; "
            "this checkout uses FastWAM's WanVideoDiT. For video generation, run the "
            "FastWAM-style sampling flow via MoT.forward."
        )
