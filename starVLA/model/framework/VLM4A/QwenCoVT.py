# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenCoVT: Qwen3-VL with Chain-of-Visual-Thought (segmentation-only).

Architecture:
    image+text  ─▶  Qwen3-VL  ─▶  hidden_states[-1]
                                         │ (positions of <|sam_pad|>)
                                         ▼
                              sam_projection (H → 256)
                                         │
                       sam_query (8×256) ◀╴cross-attn╶▶ projected hidden
                                         │
                              SAM mask_decoder (frozen)
                                         │
                              ⇣ Hungarian-match against GT masks

Loss = LM-loss (vlm_loss)  +  λ_seg · seg_loss

Stage scheduling (governed by ``self.global_step`` and the ``stage_*``
fields populated from YAML):

    step ≤ stage_warmup_steps          : LM only (no anchor token in prompt)
    step ≤ stage_align_steps           : feature-alignment only
                                         (sam_proj output ↔ SAM image embed mean)
    step ≤ stage_full_steps            : full mask reconstruction loss
    step  > stage_full_steps           : LM only (efficient inference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.visual_anchor import build_anchors
from starVLA.model.modules.visual_anchor.losses import matched_seg_loss
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

ANCHOR_START_TOKEN = "<|anchor_start|>"
ANCHOR_END_TOKEN = "<|anchor_end|>"
SAM_PAD_TOKEN = "<|sam_pad|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

NEW_SPECIAL_TOKENS = [
    ANCHOR_START_TOKEN, ANCHOR_END_TOKEN, SAM_PAD_TOKEN,
    THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE,
]


@dataclass
class QwenCoVTDefaultConfig:
    name: str = "QwenCoVT"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "Qwen/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
        }
    )

    # Visual CoT settings
    covt: dict = field(
        default_factory=lambda: {
            "num_sam_tokens": 8,
            "sam_embed_dim": 256,
            "sam_loss_weight": 1.0,
            "stage_warmup_steps": 0,         # LM only, no anchor in prompt
            "stage_align_steps": 0,          # feature-align only (skip if 0)
            "stage_full_steps": 100000,      # full mask loss until this step
            "anchor_cfg": {                  # passed to build_anchors
                "sam": {
                    "checkpoint": "./playground/Pretrained_models/sam_vit_h_4b8939.pth",
                    "model_type": "vit_h",
                    "image_size": 256,
                },
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenCoVT")
class QwenCoVT(baseframework):
    """Qwen3-VL with a segmentation-only Chain-of-Visual-Thought head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenCoVTDefaultConfig, config)
        covt_cfg = self.config.framework.covt

        # 1) Backbone VLM (Qwen3-VL-4B-Instruct by default).
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        tokenizer = self.qwen_vl_interface.processor.tokenizer

        # 2) Inject special tokens; resize embedding & lm_head accordingly.
        self._sam_token_id, self._old_vocab_size = self._add_special_tokens(tokenizer)
        self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))

        # 3) SAM mask-decoder head — projection / queries / cross-attn.
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.num_sam_tokens = int(covt_cfg.num_sam_tokens)
        self.sam_embed_dim = int(covt_cfg.sam_embed_dim)
        self.sam_loss_weight = float(covt_cfg.sam_loss_weight)

        self.sam_projection = nn.Linear(hidden_size, self.sam_embed_dim)
        self.sam_query = nn.Parameter(
            torch.randn(self.num_sam_tokens, self.sam_embed_dim) * 0.02
        )
        self.sam_cross_attn = nn.MultiheadAttention(
            embed_dim=self.sam_embed_dim, num_heads=8, batch_first=True
        )

        # 4) Frozen anchor teachers (SAM mask_decoder + prompt_encoder).
        self.anchors = build_anchors(dict(covt_cfg.anchor_cfg))

        # 5) Stage scheduling.
        self.stage_warmup_steps = int(covt_cfg.stage_warmup_steps)
        self.stage_align_steps = int(covt_cfg.stage_align_steps)
        self.stage_full_steps = int(covt_cfg.stage_full_steps)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=False)

        # Restrict embedding gradients to the newly added rows so the existing
        # vocab stays untouched (matches CoVT's row-mask hook).
        self._register_new_token_grad_mask()

    # ------------------------------------------------------------------ #
    # Special-token helpers
    # ------------------------------------------------------------------ #
    def _add_special_tokens(self, tokenizer) -> tuple[int, int]:
        old_size = len(tokenizer)
        existing = set(tokenizer.get_added_vocab().keys())
        to_add = [t for t in NEW_SPECIAL_TOKENS if t not in existing]
        if to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        sam_token_id = tokenizer.convert_tokens_to_ids(SAM_PAD_TOKEN)
        return int(sam_token_id), int(old_size)

    def _register_new_token_grad_mask(self) -> None:
        """Mask gradients on rows of embed/lm_head outside the newly added range."""
        embed = self.qwen_vl_interface.model.get_input_embeddings()
        lm_head = self.qwen_vl_interface.model.get_output_embeddings()
        new_size = embed.weight.shape[0]
        mask = torch.zeros(new_size, dtype=torch.bool)
        mask[self._old_vocab_size:new_size] = True
        self.register_buffer("_new_token_row_mask", mask, persistent=False)

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            return grad * self._new_token_row_mask.to(grad.device).view(-1, 1)

        embed.weight.register_hook(_hook)
        if lm_head is not None and lm_head.weight is not embed.weight:
            lm_head.weight.register_hook(_hook)

    # ------------------------------------------------------------------ #
    # Stage helpers
    # ------------------------------------------------------------------ #
    def current_stage(self) -> str:
        s = int(self.global_step.item())
        if s < self.stage_warmup_steps:
            return "warmup"
        if s < self.stage_align_steps:
            return "align"
        if s < self.stage_full_steps:
            return "full"
        return "efficient"

    @property
    def sam_token_id(self) -> int:
        return self._sam_token_id

    # ------------------------------------------------------------------ #
    # Forward (VLM-data only — this framework has no action head)
    # ------------------------------------------------------------------ #
    def supports_training_tag(self, tag: str) -> bool:  # noqa: D401
        return tag == "vlm"

    def forward_vlm(self, batch) -> Dict[str, torch.Tensor]:
        """Run VLM with optional CoVT mask-token supervision.

        ``batch`` keys (produced by the RoboInterCoVT dataset/collator):
            input_ids, attention_mask, labels, pixel_values, image_grid_thw,
            gt_masks  (List[Tensor]: per-sample [M, H, W] binary)
            images_pil (List[PIL.Image]: needed by SAM teacher)
        """
        gt_masks = batch.pop("gt_masks", None)
        images_pil = batch.pop("images_pil", None)

        out = self.qwen_vl_interface(
            **batch, output_hidden_states=True, return_dict=True
        )

        losses: Dict[str, torch.Tensor] = {"vlm_loss": out.loss}

        stage = self.current_stage()
        if stage in ("align", "full") and gt_masks is not None and images_pil is not None:
            seg_loss = self._compute_seg_loss(
                hidden=out.hidden_states[-1],
                input_ids=batch["input_ids"],
                images_pil=images_pil,
                gt_masks=gt_masks,
                stage=stage,
            )
            if seg_loss is not None:
                losses["seg_loss"] = seg_loss * self.sam_loss_weight

        self.global_step += 1
        return losses

    # ------------------------------------------------------------------ #
    # CoVT seg loss
    # ------------------------------------------------------------------ #
    def _compute_seg_loss(
        self,
        hidden: torch.Tensor,           # [B, T, H]
        input_ids: torch.Tensor,        # [B, T]
        images_pil: List[Image.Image],
        gt_masks: List[torch.Tensor],   # per-sample [M, H, W] binary
        stage: str,
    ) -> Optional[torch.Tensor]:
        sam_mask = input_ids == self.sam_token_id
        if not sam_mask.any():
            return None

        # Gather the (num_sam_tokens) hidden states per sample that contain anchors.
        valid_idx = [i for i in range(input_ids.size(0)) if sam_mask[i].any()]
        if not valid_idx:
            return None

        feats = []
        for i in valid_idx:
            feats.append(hidden[i, sam_mask[i]])           # [num_sam_tokens, H]
        feats = torch.stack(feats, dim=0)                  # [B', K, H]

        proj = self.sam_projection(feats)                  # [B', K, 256]
        q = self.sam_query.unsqueeze(0).expand(len(valid_idx), -1, -1).to(proj.dtype)
        token_embeds, _ = self.sam_cross_attn(q, proj, proj)   # [B', K, 256]

        if stage == "align":
            # Cheap warmup: align mean of token_embeds to SAM image embedding mean.
            with torch.no_grad():
                tgt = []
                for i in valid_idx:
                    emb = self.anchors["sam"].encode_image(images_pil[i])  # [1, 256, 64, 64]
                    tgt.append(emb.mean(dim=(2, 3)))                       # [1, 256]
                tgt = torch.cat(tgt, dim=0).to(token_embeds.dtype)         # [B', 256]
            return F.mse_loss(token_embeds.mean(dim=1), tgt)

        # Full stage: decode masks via SAM mask_decoder + Hungarian match.
        device = token_embeds.device
        per_sample_losses = []
        for offset, i in enumerate(valid_idx):
            with torch.no_grad():
                img_emb = self.anchors["sam"].encode_image(images_pil[i])
            pred = self.anchors["sam"].decode_with_tokens(
                img_emb, images_pil[i], token_embeds[offset].to(torch.float32)
            )                                                              # [K, H, W]
            gt = gt_masks[i].to(device=device, dtype=torch.float32)        # [M, H, W]
            if gt.numel() == 0:
                continue
            # Resize GT to match pred resolution if needed.
            if gt.shape[-2:] != pred.shape[-2:]:
                gt = F.interpolate(
                    gt.unsqueeze(1), size=pred.shape[-2:], mode="nearest"
                ).squeeze(1)
            per_sample_losses.append(matched_seg_loss(pred, gt))

        if not per_sample_losses:
            return None
        return torch.stack(per_sample_losses).mean()
