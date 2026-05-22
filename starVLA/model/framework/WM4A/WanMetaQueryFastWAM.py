# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""WanMetaQueryFastWAM: VLM-guided video+action joint world model.

Idea (Plan B of the design discussion):
    1. Run a frozen Qwen3-VL on the conditioning frame + instruction
       with a tail of N learnable ``<|img_query|>`` placeholder tokens.
    2. Take the last-layer hidden states at those N positions, run them
       through a MetaQuery-style transformer connector to map to the
       Wan cross-attention dim (4096, matching UMT5-XXL).
    3. *Append* these N tokens to Wan's UMT5 ``text_embeds`` (so video
       and action DiTs see ``[T5_text, VLM_queries, proprio]``).
    4. Everything downstream — MoT joint attention, video FM loss,
       action FM loss — is inherited unchanged from ``Wan_FastWAM``.

The override surface is therefore tiny: we wrap ``backbone.build_inputs``
so it returns context that already includes the VLM queries; the parent
class's ``forward`` / ``predict_action`` / ``predict_action_joint`` do
not need to be redefined.

Default freeze plan (matches MetaQuery):
    - Qwen3-VL backbone: frozen
    - newly added token embedding rows: trainable (row-mask hook)
    - MetaQuery connector: trainable (random init)
    - Wan video DiT + ActionDiT + proprio_encoder: trainable (inherit)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from starVLA.model.framework.WM4A.WanFastWAM import (
    Wan_FastWAM,
    WanFastWAMDefaultConfig,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.projector.metaquery_connector import MetaQueryConnector
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# Reserved-for-VLM special tokens (mirrors MetaQuery's BOI/EOI + <imgK>):
#     "<|begin_of_img|><|img0|><|img1|>...<|img{N-1}|><|end_of_img|>"
# Each <|imgK|> is its own special token so it has a unique embedding row.
BOI_TOKEN = "<|begin_of_img|>"
EOI_TOKEN = "<|end_of_img|>"
IMG_QUERY_TOKEN_FMT = "<|img{i}|>"


@dataclass
class WanMetaQueryFastWAMDefaultConfig(WanFastWAMDefaultConfig):
    """Same defaults as ``Wan_FastWAM`` plus a ``vlm:`` section."""

    name: str = "WanMetaQueryFastWAM"

    vlm: dict = field(
        default_factory=lambda: {
            # MLLM that produces the query hidden states.
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "sdpa",
            # Number of <|img_query|> queries appended to the VLM prompt tail.
            "num_queries": 64,
            # Whether the MLLM backbone itself is trainable (default frozen,
            # à la MetaQuery). Setting True will switch to full fine-tuning of
            # the VLM — recommended only with LoRA on top.
            "trainable": False,
            # MetaQueryConnector (Qwen3-encoder) hyperparams.
            # Per-layer shape (heads / kv_heads / ffn / RoPE) follows MetaQuery
            # exactly (`metaquery/models/model.py:203-214`):
            #   heads = hidden // 64, kv_heads = heads (no GQA), ffn = 4× hidden.
            # ``None`` triggers automatic computation from the VLM hidden size.
            #
            # Depth: we DOWNSIZE from MetaQuery's 24 → 8 because Wan video DiT
            # is fully trainable here (unlike MetaQuery, which freezes the
            # diffusion model). With a trainable downstream, the connector
            # doesn't need to do all the heavy lifting alone — 8 layers is in
            # the same ballpark as Q-Former / Perceiver Resampler. Set back to
            # 24 if you want strict MetaQuery-XL alignment.
            "connector_num_layers": 8,
            "connector_num_heads": None,        # auto: vlm_hidden // 64  (e.g. 40 for 2560)
            "connector_num_kv_heads": None,     # auto: same as num_heads (MetaQuery has no GQA)
            "connector_ffn_mult": 4,
            "connector_rope_theta": 1_000_000.0,  # Qwen3 default; MetaQuery used Qwen2 default (1e4)
            "rms_init": 1.0,
        }
    )


@FRAMEWORK_REGISTRY.register("WanMetaQueryFastWAM")
class WanMetaQueryFastWAM(Wan_FastWAM):
    """VLM-augmented variant of ``Wan_FastWAM`` (Plan B context concat)."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # First merge with our extended default so ``vlm:`` is present.
        merged_cfg = merge_framework_config(WanMetaQueryFastWAMDefaultConfig, config)
        super().__init__(config=merged_cfg)
        # ``super().__init__`` already re-merged with the parent's default —
        # but our extra ``vlm`` field is preserved verbatim (merge_framework_config
        # only adds keys that the structured default has).
        if not hasattr(self.config.framework, "vlm"):
            # Fallback for callers passing a config that doesn't carry vlm.
            self.config.framework.vlm = merged_cfg.framework.vlm

        vlm_cfg = self.config.framework.vlm

        # --------- 1. Load Qwen3-VL backbone + tokenizer ----------
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        attn_impl = vlm_cfg.get("attn_implementation", "sdpa")
        if attn_impl == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WanMetaQueryFastWAM] flash_attn not installed, falling back to sdpa")
                attn_impl = "sdpa"

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_cfg.base_vlm,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_cfg.base_vlm)
        self.vlm_processor.tokenizer.padding_side = "left"

        # --------- 2. Register query tokens & resize embedding ----------
        self.num_queries = int(vlm_cfg.num_queries)
        self._old_vocab_size = len(self.vlm_processor.tokenizer)
        self._setup_vlm_query_tokens()

        # --------- 3. Connector ----------
        vlm_text_cfg = getattr(self.vlm.config, "text_config", self.vlm.config)
        vlm_hidden = int(vlm_text_cfg.hidden_size)
        out_dim = int(self.config.framework.action_dit.text_dim)
        # Resolve auto-derived heads / kv_heads (MetaQuery formula).
        _h = vlm_cfg.get("connector_num_heads", None)
        if _h is None:
            assert vlm_hidden % 64 == 0, (
                f"vlm_hidden={vlm_hidden} is not divisible by 64; set "
                f"framework.vlm.connector_num_heads explicitly."
            )
            num_heads = vlm_hidden // 64
        else:
            num_heads = int(_h)
        _kv = vlm_cfg.get("connector_num_kv_heads", None)
        num_kv_heads = num_heads if _kv is None else int(_kv)

        self.connector = MetaQueryConnector(
            in_dim=vlm_hidden,
            out_dim=out_dim,
            num_layers=int(vlm_cfg.connector_num_layers),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_mult=int(vlm_cfg.connector_ffn_mult),
            rope_theta=float(vlm_cfg.get("connector_rope_theta", 1_000_000.0)),
            rms_init=float(vlm_cfg.rms_init),
        )

        # --------- 4. Freeze plan ----------
        if not bool(vlm_cfg.trainable):
            for p in self.vlm.parameters():
                p.requires_grad_(False)
            # But keep the *new* token rows trainable so they learn to focus
            # the VLM hidden states. Row-mask hook (CoVT pattern).
            self._register_vlm_new_token_grad_mask()

        # --------- 5. Monkey-patch backbone.build_inputs ----------
        # After this, every parent method that calls
        # ``self.backbone.build_inputs(...)`` automatically gets
        # ``wm_inputs["encoder_hidden_states"]`` with VLM queries appended.
        self._wrap_backbone_build_inputs()

        logger.info(
            f"WanMetaQueryFastWAM: VLM={vlm_cfg.base_vlm}, num_queries={self.num_queries}, "
            f"vlm_hidden={vlm_hidden}, connector_layers={vlm_cfg.connector_num_layers}, "
            f"vlm_trainable={vlm_cfg.trainable}"
        )

    # ------------------------------------------------------------------ #
    # Special-token setup
    # ------------------------------------------------------------------ #
    def _setup_vlm_query_tokens(self) -> None:
        """Add BOI/EOI + N numbered ``<|imgK|>`` tokens to the VLM tokenizer
        and resize the embedding to match (matches MetaQuery exactly)."""
        tok = self.vlm_processor.tokenizer
        img_tokens = [IMG_QUERY_TOKEN_FMT.format(i=i) for i in range(self.num_queries)]
        all_new = [BOI_TOKEN, EOI_TOKEN] + img_tokens

        existing = set(tok.get_added_vocab().keys())
        to_add = [t for t in all_new if t not in existing]
        if to_add:
            tok.add_special_tokens({"additional_special_tokens": to_add})
            self.vlm.resize_token_embeddings(len(tok))

        self.boi_token_id = int(tok.convert_tokens_to_ids(BOI_TOKEN))
        self.eoi_token_id = int(tok.convert_tokens_to_ids(EOI_TOKEN))
        self.img_query_token_ids = [
            int(tok.convert_tokens_to_ids(t)) for t in img_tokens
        ]
        # Cache the literal suffix string (used by every forward).
        self._vlm_suffix = (
            BOI_TOKEN + "".join(img_tokens) + EOI_TOKEN
        )

    def _register_vlm_new_token_grad_mask(self) -> None:
        """Allow gradient only on the newly added rows of the VLM embedding /
        lm_head. Matches CoVT's row-mask hook."""
        embed = self.vlm.get_input_embeddings()
        lm_head = self.vlm.get_output_embeddings()
        new_size = embed.weight.shape[0]
        mask = torch.zeros(new_size, dtype=torch.bool)
        mask[self._old_vocab_size:new_size] = True
        self.register_buffer("_vlm_new_token_row_mask", mask, persistent=False)

        # Unfreeze just those two parameter tensors first.
        embed.weight.requires_grad_(True)
        if lm_head is not None:
            lm_head.weight.requires_grad_(True)

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            m = self._vlm_new_token_row_mask.to(grad.device).view(-1, 1)
            return grad * m

        embed.weight.register_hook(_hook)
        if lm_head is not None and lm_head.weight is not embed.weight:
            lm_head.weight.register_hook(_hook)

    # ------------------------------------------------------------------ #
    # Optimizer integration: pull VLM embed/lm_head out into a wd=0 group
    # ------------------------------------------------------------------ #
    def extra_optimizer_groups(self) -> List[dict]:
        """Carve the VLM embed_tokens.weight and lm_head.weight tensors into a
        dedicated ``weight_decay=0`` param group.

        Rationale: ``_register_vlm_new_token_grad_mask`` installs a row-mask hook
        that zeros gradients on the 151669 pre-existing rows so the optimizer's
        grad-driven update is exactly 0 for them. But AdamW's *decoupled* weight
        decay path runs unconditionally — ``param *= (1 - lr × wd)`` — which
        would shrink every row, frozen or not, by ~3% over a 30k-step run at
        lr=1e-4, wd=1e-2. The shrink is uniform across all 151669 old rows so
        Qwen3's RMSNorm after the embed lookup largely absorbs it; lm_head's
        shrink is irrelevant because we read ``out.hidden_states[-1]`` and never
        consume the lm_head logits. But it still introduces a slow drift in the
        relative scale of the 66 trainable new rows vs the 151669 frozen old
        rows that this codebase relies on the row-mask hook to preserve, and
        the safest thing to do is just turn the decay off for both tensors.

        We DO NOT split per-row (wd=0 only on frozen rows): PyTorch optimizer
        groups operate at tensor granularity. New rows also get wd=0 here,
        which removes a small amount of regularization on the 66 trainable
        rows; acceptable trade because Adam's update magnitude on these rows
        is dominated by the actual gradient signal, not by the wd term.

        Used by ``setup_optimizer_and_scheduler`` in ``train_starvla.py``.
        """
        embed_w = self.vlm.get_input_embeddings().weight
        lm_head_module = self.vlm.get_output_embeddings()
        lm_head_w = lm_head_module.weight if lm_head_module is not None else None

        base_lr = float(self.config.trainer.learning_rate.base)
        groups = [{
            "params": [embed_w],
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "vlm.embed_tokens_no_decay",
        }]
        if lm_head_w is not None and lm_head_w is not embed_w:
            groups.append({
                "params": [lm_head_w],
                "lr": base_lr,
                "weight_decay": 0.0,
                "name": "vlm.lm_head_no_decay",
            })
        return groups

    # ------------------------------------------------------------------ #
    # build_inputs wrapping
    # ------------------------------------------------------------------ #
    def _wrap_backbone_build_inputs(self) -> None:
        """Override ``self.backbone.build_inputs`` so its return dict carries
        ``[text_embeds, vlm_queries]`` under ``encoder_hidden_states``."""

        _orig = self.backbone.build_inputs

        def _wrapped(images, instructions, image_height=None, image_width=None, **kw):
            wm_inputs = _orig(
                images, instructions,
                image_height=image_height, image_width=image_width, **kw,
            )
            vlm_ctx, vlm_mask = self._encode_vlm_context(images, instructions)
            text = wm_inputs["encoder_hidden_states"]
            text_mask = wm_inputs.get("encoder_attention_mask", None)
            if text_mask is None:
                text_mask = torch.ones(
                    text.shape[:2], dtype=torch.bool, device=text.device
                )
            wm_inputs["encoder_hidden_states"] = torch.cat(
                [text, vlm_ctx.to(text.dtype)], dim=1
            )
            wm_inputs["encoder_attention_mask"] = torch.cat(
                [text_mask, vlm_mask.to(text_mask.device)], dim=1
            )
            return wm_inputs

        # Bypass any attribute setter that backbone might have.
        object.__setattr__(self.backbone, "build_inputs", _wrapped)

    # ------------------------------------------------------------------ #
    # VLM encoding
    # ------------------------------------------------------------------ #
    def _encode_vlm_context(
        self,
        images,           # list of length B; each is List[PIL] of T frames
        instructions,     # list[str] length B
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen3-VL on (first-frame, instruction) + N query tokens.

        Returns:
            vlm_ctx:  [B, N, out_dim]   connector-mapped query embeddings
            vlm_mask: [B, N] bool        all-ones (queries are dense)
        """
        # Build per-sample messages; append the cached MetaQuery suffix
        # "<|begin_of_img|><|img0|>...<|img{N-1}|><|end_of_img|>".
        messages = []
        for img_seq, instr in zip(images, instructions):
            img = img_seq[0] if isinstance(img_seq, (list, tuple)) else img_seq
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"{instr}\n{self._vlm_suffix}"},
                ],
            }])

        inputs = self.vlm_processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.vlm.device)

        # Forward (gradients flow only through new token embedding rows + connector).
        use_grad = any(p.requires_grad for p in self.vlm.parameters())
        ctx_mgr = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx_mgr, torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.vlm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        h = out.hidden_states[-1]                       # [B, T, vlm_hidden]

        # MetaQuery-style extraction: slice every position strictly between BOI
        # and EOI. We constructed the suffix so that span is exactly N tokens
        # (one per <|imgK|>); the reshape asserts this invariant.
        input_ids = inputs["input_ids"]
        B, T = input_ids.shape
        boi_pos = (input_ids == self.boi_token_id).int().argmax(dim=1)   # [B]
        eoi_pos = (input_ids == self.eoi_token_id).int().argmax(dim=1)   # [B]
        col = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        span_mask = (col > boi_pos.unsqueeze(1)) & (col < eoi_pos.unsqueeze(1))
        if span_mask.sum().item() != B * self.num_queries:
            raise RuntimeError(
                f"BOI..EOI span produced {int(span_mask.sum().item())} tokens, "
                f"expected B*N = {B * self.num_queries}. Tokenizer may have "
                f"split / merged the <|imgK|> tokens."
            )
        q_hidden = h[span_mask].view(B, self.num_queries, h.size(-1))    # [B, N, vlm_hidden]

        vlm_ctx = self.connector(q_hidden)               # [B, N, out_dim]
        vlm_mask = torch.ones(
            (B, self.num_queries), dtype=torch.bool, device=vlm_ctx.device
        )
        return vlm_ctx, vlm_mask
