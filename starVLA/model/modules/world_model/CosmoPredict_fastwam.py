# Copyright 2025 starVLA community. All rights reserved.
"""Cosmos-Predict2 World Model Interface — FastWAM-style backbone for joint MoT.

Drop-in alternative to ``Wan2_fastwam.WanVideoBackboneFastWAM`` that swaps the
Wan2.2-TI2V-5B DiT for NVIDIA Cosmos-Predict2-2B-Video2World. Same public API
(``build_inputs``, ``_encode_text``, ``_encode_images_vae``, ``forward``,
``generate``) so the WanFastWAM framework can route between Wan and Cosmos
backbones with a yaml flag.

Module choices (vs Wan2_fastwam):
  | Component       | Wan2_fastwam (FastWAM)              | This file (Cosmos)              |
  | --------------- | ----------------------------------- | --------------------------------|
  | VAE             | DiffSynth WanVideoVAE38 (vendored)  | diffusers AutoencoderKLWan      |
  | Text encoder    | DiffSynth WanTextEncoder (UMT5-XXL) | transformers T5EncoderModel     |
  | Tokenizer       | FastWAM HuggingfaceTokenizer        | T5TokenizerFast                 |
  | Scheduler       | WanContinuousFlowMatchScheduler     | FlowMatchEulerDiscreteScheduler |
  | Video DiT       | FastWAM WanVideoDiT (30 layers,     | diffusers CosmosTransformer3D   |
  |                 |  hidden=3072, 24h × 128 = 3072)     |  (28 layers, hidden=2048,       |
  |                 |                                     |  16h × 128 = 2048)              |
  | text_dim        | 4096 (UMT5-XXL)                     | 1024 (T5-XXL — different size!) |

The **text_dim mismatch** (4096 vs 1024) is the critical interface gap with
WanFastWAM: ActionDiT defaults to text_dim=4096 to match Wan's UMT5. For Cosmos,
either (a) project T5 1024→4096 with a learnable Linear, or (b) configure
ActionDiT with text_dim=1024. The yaml must pick one; this file just produces
1024-dim text embeds and lets the framework decide.

Joint MoT integration (TODO):
  ``WanVideoDiT`` exposes ``pre_dit``, ``post_dit``, and ``blocks`` as the
  interface MoT consumes (see ``mot.py``). CosmosTransformer3DModel from
  diffusers does NOT expose these — its forward is monolithic. To fully wire
  Cosmos into MoT we need one of:

    (a) Fork CosmosTransformer3DModel, split forward into pre_dit + blocks +
        post_dit. Adapter then mirrors the FastWAM split.
    (b) Replace CosmosTransformer3DModel with a re-implementation that exposes
        the split, loading weights from the diffusers state dict.
    (c) Accept lower-fidelity integration: use Cosmos transformer as a feature
        extractor only (run the full forward, capture hidden states via hooks),
        then route to ActionDiT via cross-attention (NOT joint MoT). This is
        what ``CosmoPredict2GR00T.py`` already does.

This file goes with (a) as the long-term plan but ships option (c) wiring
today: ``pre_dit`` / ``post_dit`` raise ``NotImplementedError`` so the
framework knows joint MoT is not yet supported on Cosmos. The framework can
fall back to feature-extraction mode via a config flag.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Cosmos-Predict2-2B-Video2World repo on HuggingFace — same default as
# CosmoPredict2.py. Override via yaml world_model.base_wm.
_DEFAULT_MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"


class _TransformerConfigShim:
    """Mirror ``Wan2_fastwam._TransformerConfigShim``: framework code reads
    ``transformer.config.{patch_size, hidden_dim, num_heads, ...}``. Cosmos's
    diffusers config exposes these under different names, so we proxy."""

    def __init__(self, cosmos_cfg):
        # CosmosTransformer3DModel attributes (snake_case from diffusers)
        # Map to the names WanFastWAM framework expects
        self.patch_size = tuple(cosmos_cfg.get("patch_size", (1, 2, 2)))
        self.hidden_dim = int(
            cosmos_cfg.get("num_attention_heads", 16)
            * cosmos_cfg.get("attention_head_dim", 128)
        )
        self.num_heads = int(cosmos_cfg.get("num_attention_heads", 16))
        self.attn_head_dim = int(cosmos_cfg.get("attention_head_dim", 128))
        self.num_layers = int(cosmos_cfg.get("num_layers", 28))
        self.in_dim = int(cosmos_cfg.get("in_channels", 16) + 1)  # +1 for condition_mask
        self.out_dim = int(cosmos_cfg.get("out_channels", 16))
        self.text_dim = int(cosmos_cfg.get("text_embed_dim", 1024))


class WanVideoBackboneFastWAMCosmos(nn.Module):
    """Cosmos-Predict2 backbone with FastWAM-aligned IO interface.

    Identical public surface to ``Wan2_fastwam.WanVideoBackboneFastWAM``:
        - ``build_inputs(images, instructions, ...)`` → dict with
          ``hidden_states`` (encoded video latents), ``encoder_hidden_states``
          (text embeds), ``encoder_attention_mask``, ``timestep``,
          ``_is_wm_input``.
        - ``_encode_text(instructions)``  — T5 text encoding (or cache lookup).
        - ``_encode_images_vae(images, ...)`` — VAE encode video.
        - ``forward(**kwargs)`` — raises (use framework MoT path instead).
        - ``generate(**kwargs)`` — raises.

    The transformer is exposed at ``self.transformer`` so framework code can
    reach into ``self.transformer.config`` for geometry queries. The
    transformer's ``blocks`` attribute is aliased onto ``self.transformer.blocks``
    so ``len(blocks)`` queries work; per-block num_heads / attn_head_dim are
    exposed via ``_TransformerConfigShim`` for the MoT consistency checks.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get("base_wm", _DEFAULT_MODEL)
        self.config = config

        from diffusers import (
            AutoencoderKLWan,
            CosmosTransformer3DModel,
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import T5EncoderModel, T5TokenizerFast

        from .cosmos_video_dit import CosmosVideoDiT

        logger.info(f"[CosmoPredict_fastwam] Loading Cosmos-Predict2 from {model_name}")

        # --- Tokenizer + Text encoder (T5XXL, 1024-dim) -----------------
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        # Allow text_embed cache to bypass loading the (heavy) T5 weights — same
        # as Wan2_fastwam's load_text_encoder=False mode.
        load_text_encoder = bool(wm_cfg.get("load_text_encoder", True))
        if load_text_encoder:
            self.text_encoder = T5EncoderModel.from_pretrained(
                model_name,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
            )
            self.text_encoder.requires_grad_(False)
        else:
            self.text_encoder = None

        # --- Transformer ------------------------------------------------
        # We instantiate our FastWAM-style ``CosmosVideoDiT`` and try to
        # copy weights from the diffusers ``CosmosTransformer3DModel`` so
        # the model has a sensible init. The diffusers transformer is then
        # released to free memory.
        skip_dit_load = bool(wm_cfg.get("skip_transformer_load", False))
        if skip_dit_load:
            logger.info("[CosmoPredict_fastwam] Skipping diffusers DiT load (skip_transformer_load=True)")
            diffusers_state = None
            # Read transformer config from a partial load to get shape params.
            diffusers_transformer = CosmosTransformer3DModel.from_pretrained(
                model_name, subfolder="transformer",
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
            orig_cfg = diffusers_transformer.config
            del diffusers_transformer
        else:
            diffusers_transformer = CosmosTransformer3DModel.from_pretrained(
                model_name, subfolder="transformer", torch_dtype=torch.bfloat16
            )
            diffusers_state = diffusers_transformer.state_dict()
            orig_cfg = diffusers_transformer.config
            del diffusers_transformer  # free memory after grabbing state

        cfg_dict = {
            "patch_size": tuple(getattr(orig_cfg, "patch_size", (1, 2, 2))),
            "num_attention_heads": int(orig_cfg.num_attention_heads),
            "attention_head_dim": int(orig_cfg.attention_head_dim),
            "num_layers": int(orig_cfg.num_layers),
            "in_channels": int(orig_cfg.in_channels),
            "out_channels": int(orig_cfg.out_channels),
            "text_embed_dim": int(getattr(orig_cfg, "text_embed_dim", 1024)),
            "mlp_ratio": float(getattr(orig_cfg, "mlp_ratio", 4.0)),
            "max_size": tuple(getattr(orig_cfg, "max_size", (128, 240, 240))),
            "rope_scale": tuple(getattr(orig_cfg, "rope_scale", (2.0, 1.0, 1.0))),
            "concat_padding_mask": bool(getattr(orig_cfg, "concat_padding_mask", True)),
        }
        hidden = cfg_dict["num_attention_heads"] * cfg_dict["attention_head_dim"]
        ffn_dim = int(hidden * cfg_dict["mlp_ratio"])

        # Diffusers config's ``in_channels`` already includes the condition_mask
        # slot (z_dim=16 + condition_mask=1 = 17 in Cosmos-Predict2). The
        # patch_embed Conv3d input is then ``in_channels + 1`` when
        # concat_padding_mask=True. So our in_dim = cfg.in_channels + (1 if pad).
        in_dim = cfg_dict["in_channels"]
        if cfg_dict["concat_padding_mask"]:
            in_dim += 1
        self._z_dim = cfg_dict["in_channels"] - 1  # latent channels only (subtract condition_mask slot)

        self.transformer = CosmosVideoDiT(
            hidden_dim=hidden,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=cfg_dict["out_channels"],
            text_dim=cfg_dict["text_embed_dim"],
            patch_size=cfg_dict["patch_size"],
            num_heads=cfg_dict["num_attention_heads"],
            attn_head_dim=cfg_dict["attention_head_dim"],
            num_layers=cfg_dict["num_layers"],
            max_size=cfg_dict["max_size"],
            rope_scale=cfg_dict["rope_scale"],
            seperated_timestep=True,
            fuse_vae_embedding_in_latents=True,
        ).to(torch.bfloat16)

        # Load what we can from diffusers state_dict (weights for attn / ffn
        # / head linears; modulation params and norms get random init — see
        # CosmosVideoDiT.load_from_diffusers_state_dict docstring).
        if diffusers_state is not None:
            report = self.transformer.load_from_diffusers_state_dict(diffusers_state, strict=False)
            logger.info(f"[CosmoPredict_fastwam] diffusers→CosmosVideoDiT remap: "
                        f"{len(report['loaded'])} loaded, {len(report['missing'])} missing")
            del diffusers_state

        # Re-shim transformer.config for framework-side attribute reads.
        self.transformer.config = _TransformerConfigShim(cfg_dict)
        self._concat_padding_mask = cfg_dict["concat_padding_mask"]

        # --- VAE (AutoencoderKLWan, same as Wan but loaded via diffusers) -
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.vae.requires_grad_(False)

        # --- Scheduler (kept for compat parity; framework owns its own) ---
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # --- VAE scale factors --------------------------------------------
        # Cosmos VAE: temperal_downsample is a list [True, True] for 4x temporal.
        # Same shape contract as Wan2.py/Wan2_fastwam.py.
        from diffusers.video_processor import VideoProcessor
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # --- Text-embed cache (same support as Wan2_fastwam) --------------
        self._text_prompt_template = wm_cfg.get("text_prompt_template", None)
        self._zero_pad_text_embeds = bool(wm_cfg.get("zero_pad_text_embeds", False))
        self._force_text_mask_ones = bool(wm_cfg.get("force_text_mask_ones", False))
        self._text_embed_cache = None
        self._text_embed_cache_dir = None
        self._text_embed_max_length = int(wm_cfg.get("text_embed_max_length", 512))
        cache_path = wm_cfg.get("text_embed_cache_path", None)
        if cache_path:
            cache_obj = Path(str(cache_path))
            if not cache_obj.exists():
                raise FileNotFoundError(f"text_embed_cache_path={cache_path} does not exist")
            if cache_obj.is_dir():
                logger.info(f"[CosmoPredict_fastwam] Using text cache dir {cache_path}")
                self._text_embed_cache_dir = str(cache_obj)
                self.text_encoder = None
                self.tokenizer = None
            else:
                payload = torch.load(str(cache_obj), map_location="cpu", weights_only=False)
                self._text_embed_cache = payload["cache"]
                self._text_embed_max_length = int(payload.get("max_length", 512))
                self.text_encoder = None
                self.tokenizer = None

        # Hidden dim accessor used by some framework codepaths.
        self._hidden_size = (
            int(cfg_dict["num_attention_heads"]) * int(cfg_dict["attention_head_dim"])
        )
        logger.info(
            f"[CosmoPredict_fastwam] transformer: layers={cfg_dict['num_layers']} "
            f"hidden={self._hidden_size} text_dim={cfg_dict['text_embed_dim']} "
            f"(NOTE: text_dim != 4096 → ActionDiT.text_dim must match in yaml)"
        )

    # ----------------------------------------------------------
    # Text encoding (parity with Wan2_fastwam)
    # ----------------------------------------------------------
    def _apply_text_template(self, instruction: str) -> str:
        if not self._text_prompt_template:
            return instruction
        return self._text_prompt_template.format(task=instruction)

    @torch.no_grad()
    def _encode_text(self, instructions, max_length: Optional[int] = None):
        """Encode text instructions → (text_embeds, text_attention_mask).

        Modes (same as Wan2_fastwam._encode_text):
          1. ``self._text_embed_cache`` is a dict {instruction: tensor} →
             lookup, no T5 call.
          2. ``self._text_embed_cache_dir`` is a directory of per-instruction
             .safetensors → file lookup.
          3. ``self.tokenizer`` + ``self.text_encoder`` present → run T5.

        Returns shape ``[B, L, text_dim]`` and mask ``[B, L]``.
        """
        if max_length is None:
            max_length = self._text_embed_max_length

        # Apply template (e.g. "A video... {task}").
        rendered = [self._apply_text_template(s) for s in instructions]

        device = next(self.parameters()).device

        # --- Mode 1: in-memory dict cache -------------------------------
        if self._text_embed_cache is not None:
            embeds = []
            masks = []
            for s in rendered:
                if s not in self._text_embed_cache:
                    raise KeyError(
                        f"text_embed_cache missing key {s!r}. Either regenerate "
                        f"the cache with the current template or disable it."
                    )
                e = self._text_embed_cache[s].to(device=device, dtype=torch.bfloat16)
                if e.dim() == 2:
                    e = e.unsqueeze(0)
                embeds.append(e)
                # Cache only stores the embeds, mask = ones (assumes pre-trimmed)
                masks.append(torch.ones(e.shape[1], dtype=torch.bool, device=device))
            text_embeds = torch.stack([e.squeeze(0) for e in embeds], dim=0)
            text_mask = torch.stack(masks, dim=0)
            return text_embeds, text_mask

        # --- Mode 2: per-instruction safetensors directory --------------
        if self._text_embed_cache_dir is not None:
            from safetensors.torch import load_file
            cache_dir = Path(self._text_embed_cache_dir)
            embeds = []
            masks = []
            for s in rendered:
                # FastWAM-style: cache key is the rendered string hashed/filename-safe
                # — but the typical convention is filename = sanitized instruction.
                # We mirror the Wan2_fastwam path: filename = instruction with
                # non-safe chars replaced. Caller is responsible for matching.
                safe = "".join(c if c.isalnum() else "_" for c in s)[:200]
                fpath = cache_dir / f"{safe}.safetensors"
                if not fpath.exists():
                    raise FileNotFoundError(
                        f"Cache miss for instruction {s!r} → looked at {fpath}. "
                        f"Pre-generate with scripts/precompute_libero_text_embeds.py "
                        f"(adapt for Cosmos T5XXL with text_dim=1024)."
                    )
                payload = load_file(str(fpath))
                e = payload["text_embeds"].to(device=device, dtype=torch.bfloat16)
                m = payload.get("text_mask", torch.ones(e.shape[0], dtype=torch.bool))
                embeds.append(e)
                masks.append(m.to(device=device, dtype=torch.bool))
            text_embeds = torch.stack(embeds, dim=0)
            text_mask = torch.stack(masks, dim=0)
            return text_embeds, text_mask

        # --- Mode 3: live T5 encode -------------------------------------
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError(
                "Text encoder + tokenizer were unloaded but no cache was supplied."
            )
        tokens = self.tokenizer(
            rendered,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = self.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )
        text_embeds = out.last_hidden_state.to(dtype=torch.bfloat16)
        text_mask = tokens.attention_mask.to(dtype=torch.bool)
        if self._zero_pad_text_embeds:
            # Same trick as Wan2_fastwam: mask out PAD tokens with zeros so
            # downstream cross-attention has well-defined attended values even
            # if the mask is ignored.
            text_embeds = text_embeds * text_mask.unsqueeze(-1)
        if self._force_text_mask_ones:
            text_mask = torch.ones_like(text_mask)
        return text_embeds, text_mask

    # ----------------------------------------------------------
    # Image / video VAE encoding (parity with Wan2_fastwam)
    # ----------------------------------------------------------
    def _encode_images_vae(self, images, num_frames=None, image_height=None, image_width=None):
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        # Wan training resolution is 480×832; FastWAM uses 224×448. Default here
        # mirrors Wan2_fastwam (which the framework yaml overrides).
        height = int(image_height) if image_height is not None else 224
        width = int(image_width) if image_width is not None else 448

        preprocessed = []
        frame_counts = []
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

        # Normalize: (latents - mean) / std * sigma_data    — Cosmos pipeline.
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        sigma_data = float(getattr(self.scheduler.config, "sigma_data", 1.0))
        latents = (latents - latents_mean) / latents_std * sigma_data
        return latents

    # ----------------------------------------------------------
    # build_inputs (parity with Wan2_fastwam.build_inputs)
    # ----------------------------------------------------------
    def build_inputs(self, images, instructions, image_height=None, image_width=None, **kwargs):
        assert len(images) == len(instructions)
        text_embeds, text_mask = self._encode_text(instructions)
        latents = self._encode_images_vae(
            images, image_height=image_height, image_width=image_width
        )
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

    # ----------------------------------------------------------
    # Forward / generate — same contract as Wan2_fastwam
    # ----------------------------------------------------------
    def forward(self, **kwargs):
        # Same as Wan2_fastwam: framework code uses MoT (which interleaves
        # action expert into the video DiT's attention), so this monolithic
        # forward is intentionally not used. Calling it would bypass MoT.
        raise NotImplementedError(
            "CosmoPredict_fastwam.transformer is wrapped for joint MoT. Use the "
            "WanFastWAM-equivalent framework forward path. If you need a "
            "standalone DiT forward (e.g. for feature extraction without MoT), "
            "use CosmoPredict2.WanVideoBackbone instead."
        )

    def generate(self, **kwargs):
        raise NotImplementedError(
            "Use diffusers Cosmos2VideoToWorldPipeline directly if you need "
            "video generation. The framework code does not call this."
        )


__all__ = ["WanVideoBackboneFastWAMCosmos"]
