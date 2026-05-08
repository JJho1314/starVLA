# Copyright 2025 starVLA community.
"""FastWAM-faithful ActionDiT for starVLA.

A 30-layer DiT (self-attn + FFN + AdaLN modulation + RoPE) acting as the action
expert in a true MoT setup. Architecture is a 1:1 reproduction of FastWAM's
`fastwam/models/wan22/action_dit.py` (and the supporting `DiTBlock` from
FastWAM's `wan_video_dit.py`).

Why we need this:
- The current LayerwiseFlowmatchingActionHead is 30 cross-attention layers (action
  reads backbone hidden as K/V; action does NOT contribute its own K/V).
- FastWAM's MoT requires action to produce its own Q/K/V at each layer so that
  the joint attention can `concat([video_KV, action_KV])`. ActionDiT below
  provides exactly that interface.

This file is purely additive: WanFastWAM.py is not modified by this step. The
class will be wired into the framework in a later step (B4).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ----------------------------------------------------------------------------
# Helpers (from FastWAM wan_video_dit.py)
# ----------------------------------------------------------------------------


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: Optional[torch.Tensor] = None,
    compatibility_mode: bool = True,
) -> torch.Tensor:
    """Multi-head scaled-dot-product attention. `q/k/v` shape: [B, S, H*Dh]."""
    if not compatibility_mode:
        raise NotImplementedError(
            "Only compatibility mode is implemented for flash_attention."
        )
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    """1-D RoPE precompute. Returns complex64 tensor [end, dim//2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def gradient_checkpoint_forward(
    model: nn.Module,
    use_gradient_checkpointing: bool,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing:
        def _custom_forward(*inputs, **kw):
            return model(*inputs, **kw)
        return torch.utils.checkpoint.checkpoint(
            _custom_forward, *args, **kwargs, use_reentrant=False
        )
    return model(*args, **kwargs)


# ----------------------------------------------------------------------------
# Attention modules
# ----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self._norm(x.float()).to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual


# ----------------------------------------------------------------------------
# DiTBlock — used identically by FastWAM video and action experts
# ----------------------------------------------------------------------------


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim ** 0.5)
        self.gate = GateModule()

    def _split_modulation(self, t_mod: torch.Tensor):
        """Split (modulation + t_mod) into 6 AdaLN params. Returns 6 tensors."""
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def compute_qkv(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
    ) -> dict:
        """Pre-attention step: norm1 + AdaLN modulation + Q/K/V projection + RoPE.

        Used by both `forward` (drives full block) and B7 MoTAttnProcessor (joint
        attention path). Returns a dict containing flat-head Q/K/V tensors and
        the residual-path state needed by `post_attention`.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(t_mod)

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        q = self.self_attn.norm_q(self.self_attn.q(input_x))
        k = self.self_attn.norm_k(self.self_attn.k(input_x))
        v = self.self_attn.v(input_x)
        q = rope_apply(q, freqs, self.self_attn.num_heads)
        k = rope_apply(k, freqs, self.self_attn.num_heads)
        return {
            "q": q,
            "k": k,
            "v": v,
            "residual_x": x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }

    def post_attention(
        self,
        attn_out: torch.Tensor,
        qkv_state: dict,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Post-attention step: o-proj + gate residual + cross-attn(text) + FFN.

        `attn_out` is the joint/self attention output (flat-head shape, matching
        what `self.self_attn.o` expects: [B, S, H*D]).
        """
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)  # (B, 1, seq, ctx_len)

        residual_x = qkv_state["residual_x"]
        gate_msa = qkv_state["gate_msa"]
        shift_mlp = qkv_state["shift_mlp"]
        scale_mlp = qkv_state["scale_mlp"]
        gate_mlp = qkv_state["gate_mlp"]

        x = self.gate(residual_x, gate_msa, self.self_attn.o(attn_out))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standalone block forward: compute_qkv -> self self-attn -> post_attention.

        Behavior is identical to the previous monolithic forward. Used by
        ActionDiT.forward when running action-only (no joint MoT). The B7
        MoTAttnProcessor calls compute_qkv and post_attention directly instead.
        """
        qkv_state = self.compute_qkv(x, t_mod, freqs)
        attn_out = flash_attention(
            q=qkv_state["q"],
            k=qkv_state["k"],
            v=qkv_state["v"],
            num_heads=self.self_attn.num_heads,
            ctx_mask=self_attn_mask,
        )
        return self.post_attention(attn_out, qkv_state, context, context_mask=context_mask)


# ----------------------------------------------------------------------------
# ActionDiT — top-level action expert
# ----------------------------------------------------------------------------


class ActionDiT(nn.Module):
    """30-layer DiT acting as the action expert in true MoT.

    Defaults match FastWAM `configs/model/fastwam.yaml::action_dit_config`:
        hidden_dim=1024, ffn_dim=4096, num_heads=24, attn_head_dim=128,
        num_layers=30, text_dim=4096, freq_dim=256, eps=1e-6.
    """

    ACTION_BACKBONE_SKIP_PREFIXES: Tuple[str, ...] = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS: Tuple[str, ...] = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def backbone_key_set(cls, keys) -> set:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: Dict[str, Any],
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path`, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)

        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            # Resolve relative to starVLA repo root: this file is at
            # starVLA/model/modules/action_model/FastWAM_ActionDiT.py — go up 4 levels.
            p = Path(__file__).resolve().parents[3] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )

        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta", {})
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got {tuple(context.shape)}")

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch action vs text: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(
                    f"`context_mask` must be 2D [B, L], got {tuple(context_mask.shape)}"
                )
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape {tuple(context_mask.shape)} does not match "
                    f"`context` shape {tuple(context.shape)} on (B, L)."
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        ctx = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        ctx_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    ctx,
                    t_mod,
                    freqs,
                    context_mask=ctx_mask,
                )
            else:
                x = block(x, ctx, t_mod, freqs, context_mask=ctx_mask)

        return self.post_dit(x, pre_state)
