# Copyright 2025 starVLA community.
"""MoT-Lite: bidirectional cross-attention between Wan2.2 video tokens and ActionDiT tokens.

True FastWAM MoT does **joint** attention: at each layer, Q/K/V of video and action are
concatenated and one shared self-attention runs over the union. That requires forking the
Wan2.2 transformer block (modulation + RoPE + AttnProcessor are tightly coupled in
diffusers' `WanTransformerBlock`).

MoT-Lite achieves equivalent **information flow** with two additional cross-attentions per
layer, attached via forward post-hooks on each Wan22 block — without modifying diffusers:

    h_v ← Wan22Block(h_v)                          # original block forward (untouched)
    h_v ← h_v + γ_v * CrossAttn_VA(Q=h_v, K,V=h_a) # NEW: video reads action
    h_a ← MoTActionBlock(h_a, h_v_kv)              # NEW: action reads video + own FFN

The two cross-attentions are *small* (single layer per block, hidden_dim=1024 for action,
projected from 3072 video). γ_v is initialised at 0 so the perturbation to Wan22's output
is zero at init — backbone's video-gen prior is preserved early in training.

This sacrifices the "joint K/V pool" property of true MoT (where Q_v can attend to a single
unified K/V). For ~8 action tokens vs ~hundreds of video tokens, the practical benefit gap
is small.

Wiring (done by `WanFastWAM_MoT` framework):
    1. Construct ActionDiT (one MoTActionBlock per Wan22 block).
    2. For block_i in backbone.transformer.blocks:
         block_i.register_forward_hook(MoTLiteHook(action_block_i, parent_model))
    3. Before backbone.forward, set parent_model._action_tokens = init_query.
    4. After backbone.forward, parent_model._action_tokens has been updated through all
       layers via the hooks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Multi-head SDPA. q/k/v: [B, N, H*D]. Returns [B, N, H*D]."""
    B, Nq, _ = q.shape
    D_total = q.shape[-1]
    D_head = D_total // num_heads
    q = q.view(B, Nq, num_heads, D_head).transpose(1, 2)  # [B, H, Nq, D]
    k = k.view(B, k.shape[1], num_heads, D_head).transpose(1, 2)
    v = v.view(B, v.shape[1], num_heads, D_head).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)  # [B, H, Nq, D]
    out = out.transpose(1, 2).reshape(B, Nq, D_total)
    return out


class _ProjQKV(nn.Module):
    """Tiny QKV projection module used by MoTLite cross-attentions."""

    def __init__(self, q_dim: int, kv_dim: int, attn_dim: int):
        super().__init__()
        self.to_q = nn.Linear(q_dim, attn_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, attn_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, attn_dim, bias=False)


class MoTActionBlock(nn.Module):
    """One ActionDiT block: action reads its own state + video K/V, then FFN.

    Mirrors a Wan22 block in spirit but lighter:
        h_a ← h_a + α * CrossAttn(Q=norm(h_a), K,V=cat(norm(h_a), h_v))
        h_a ← h_a + α * FFN(norm(h_a))

    Where α gates start near zero so early training behaves like WanPI (action queries
    just read video via a thin cross-attention path; no aggressive perturbation of the
    bare Wan22 representations).

    Args:
        action_dim: action expert hidden size (e.g. 1024).
        video_dim:  Wan22 hidden size (e.g. 3072).
        num_heads:  attention heads inside the block.
        attn_head_dim: per-head dim. attn_dim = num_heads * attn_head_dim.
        ffn_dim:    FFN inner dim.
    """

    def __init__(
        self,
        action_dim: int = 1024,
        video_dim: int = 3072,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        ffn_dim: int = 4096,
    ) -> None:
        super().__init__()
        attn_dim = num_heads * attn_head_dim
        self.num_heads = num_heads
        self.action_dim = action_dim

        self.norm1 = nn.LayerNorm(action_dim)
        # action Q/K/V projects from action_dim → attn_dim
        # video K/V projects from video_dim → attn_dim
        self.qkv = _ProjQKV(q_dim=action_dim, kv_dim=action_dim, attn_dim=attn_dim)
        self.video_to_kv = _ProjQKV(q_dim=action_dim, kv_dim=video_dim, attn_dim=attn_dim)
        self.to_out = nn.Linear(attn_dim, action_dim, bias=False)

        self.norm2 = nn.LayerNorm(action_dim)
        self.ffn = nn.Sequential(
            nn.Linear(action_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, action_dim),
        )
        # zero-init scalar gate so block is identity at init
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, h_a: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        """h_a [B, N_a, action_dim], h_v [B, N_v, video_dim] → h_a' [B, N_a, action_dim]."""
        # 1. Bidirectional self+cross attention: action queries see (action + video)
        x = self.norm1(h_a)
        q = self.qkv.to_q(x)                               # [B, N_a, attn_dim]
        k_a = self.qkv.to_k(x)
        v_a = self.qkv.to_v(x)
        k_v = self.video_to_kv.to_k(h_v)                   # [B, N_v, attn_dim]
        v_v = self.video_to_kv.to_v(h_v)
        K = torch.cat([k_a, k_v], dim=1)                   # joint K
        V = torch.cat([v_a, v_v], dim=1)
        attn_out = _scaled_dot_product_attention(q, K, V, self.num_heads)
        attn_out = self.to_out(attn_out)
        h_a = h_a + self.gate_attn * attn_out

        # 2. FFN
        h_a = h_a + self.gate_ffn * self.ffn(self.norm2(h_a))
        return h_a


class _ReverseVideoUpdate(nn.Module):
    """Reverse direction: video tokens read action tokens via cross-attention.

    h_v_new = h_v + γ * CrossAttn(Q=h_v, K=h_a, V=h_a) where γ starts at 0.

    This is the "MoT-symmetry" piece — without it, action sees video but video doesn't
    see action (= WanPI). Zero-init keeps backbone outputs intact at training start.
    """

    def __init__(self, video_dim: int = 3072, action_dim: int = 1024, num_heads: int = 24):
        super().__init__()
        attn_head_dim = video_dim // num_heads
        attn_dim = num_heads * attn_head_dim   # = video_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(video_dim)
        self.to_q = nn.Linear(video_dim, attn_dim, bias=False)
        self.to_k = nn.Linear(action_dim, attn_dim, bias=False)
        self.to_v = nn.Linear(action_dim, attn_dim, bias=False)
        self.to_out = nn.Linear(attn_dim, video_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h_v: torch.Tensor, h_a: torch.Tensor) -> torch.Tensor:
        x = self.norm(h_v)
        q = self.to_q(x)
        k = self.to_k(h_a)
        v = self.to_v(h_a)
        attn_out = _scaled_dot_product_attention(q, k, v, self.num_heads)
        attn_out = self.to_out(attn_out)
        return h_v + self.gate * attn_out


class MoTLiteHook:
    """Forward post-hook attached to each Wan22 block.

    On every Wan22Block forward:
      1. read parent_model._action_tokens (state propagated layer-to-layer)
      2. reverse update: h_v ← h_v + reverse_attn(h_v, h_a)        [optional]
      3. update h_a via MoTActionBlock(h_a, h_v_post_reverse)
      4. write back parent_model._action_tokens

    Args:
        action_block:   the MoTActionBlock owned by the framework (one per layer).
        reverse_update: optional _ReverseVideoUpdate; if None, no reverse path.
        parent_attr_name: attribute on parent model holding action_tokens, default "_action_tokens".
    """

    def __init__(
        self,
        action_block: MoTActionBlock,
        reverse_update: Optional[_ReverseVideoUpdate] = None,
        parent_model: Optional[nn.Module] = None,
        parent_attr_name: str = "_action_tokens",
    ):
        self.action_block = action_block
        self.reverse_update = reverse_update
        self.parent_model = parent_model
        self.parent_attr = parent_attr_name

    def __call__(self, module, inputs, output):
        # diffusers WanTransformerBlock returns a single tensor (hidden_states)
        h_v = output if not isinstance(output, tuple) else output[0]
        h_a = getattr(self.parent_model, self.parent_attr, None)
        if h_a is None:
            # Eval / inference path may want to skip MoT (no action tokens registered)
            return output

        # Optional: video reads action (zero-init gate keeps backbone intact early)
        if self.reverse_update is not None:
            h_v = self.reverse_update(h_v, h_a)

        # Action reads (own + video) and FFN
        h_a_new = self.action_block(h_a, h_v)
        setattr(self.parent_model, self.parent_attr, h_a_new)

        if isinstance(output, tuple):
            return (h_v,) + tuple(output[1:])
        return h_v
