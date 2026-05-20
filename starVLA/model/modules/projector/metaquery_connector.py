"""MetaQuery-style connector (Qwen3-encoder variant).

Maps ``[B, N, in_dim]`` MLLM query hidden states to ``[B, N, out_dim]``
ready to be concatenated into a Wan / Sana / SD cross-attention context.

Structure (MetaQuery-aligned, using Qwen3 building blocks):
    Qwen3Encoder  (N stacked Qwen3DecoderLayer with is_causal=False)
        ↓
    Linear(in_dim → out_dim) → GELU(tanh) → Linear(out_dim → out_dim)
        ↓
    RMSNorm(out_dim)         (initialised to ``rms_init``)

Compared to MetaQuery's official Qwen2 encoder, we:
    - port the same trick (bidirectional Qwen-decoder-as-encoder via
      ``self_attn.is_causal = False``)
    - swap Qwen2 → Qwen3 (Qwen3 has built-in q_norm/k_norm by design;
      no separate ``qk_norm`` toggle needed)
    - reuse stock HF blocks ``Qwen3DecoderLayer / Qwen3RotaryEmbedding /
      Qwen3RMSNorm`` so the implementation stays small and
      torch-/HF-compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


class _RMSNorm(nn.Module):
    """Plain RMSNorm with a learned scale (no bias).

    Used at the tail of the connector (matches MetaQuery's final scale-up
    of the cross-attention context). PyTorch ``nn.RMSNorm`` only exists
    from 2.4 onward — we ship our own so starVLA's pinned torch range
    stays unrestricted.
    """

    def __init__(self, dim: int, eps: float = 1e-5, init_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), float(init_scale)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps).to(x.dtype)
        return x_norm * self.weight.to(x.dtype)


class Qwen3Encoder(nn.Module):
    """Bidirectional encoder built from stock ``Qwen3DecoderLayer`` blocks.

    The only deviation from a vanilla Qwen3 decoder stack is that each
    layer's ``self_attn.is_causal`` is set to ``False`` so the attention
    pattern becomes fully bidirectional (queries attend to every other
    query). This mirrors MetaQuery's ``Qwen2BidirectionalSdpaAttention``
    trick.

    Args:
        hidden_size:   residual stream dim of every layer.
        num_layers:    stack depth.
        num_heads:     attention heads.
        num_kv_heads:  GQA group count (defaults to ``num_heads`` if None).
        head_dim:      per-head dim (defaults to ``hidden_size//num_heads``).
        ffn_dim:       FFN intermediate dim (defaults to ``hidden_size*4``).
        rope_theta:    RoPE base frequency.
        rms_norm_eps:  RMSNorm epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 8,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        ffn_dim: int | None = None,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            assert hidden_size % num_heads == 0
            head_dim = hidden_size // num_heads
        if ffn_dim is None:
            ffn_dim = hidden_size * 4

        # Stub Qwen3Config with just the fields Qwen3DecoderLayer reads.
        cfg = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=8192,
            rope_theta=rope_theta,
            attention_dropout=0.0,
            rms_norm_eps=rms_norm_eps,
            attention_bias=False,
            sliding_window=None,
            tie_word_embeddings=False,
        )
        # SDPA path (works under DeepSpeed ZeRO and bf16 autocast).
        cfg._attn_implementation = "sdpa"

        self.config = cfg
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg, layer_idx=i) for i in range(num_layers)]
        )
        # Flip every attention from causal to bidirectional in-place.
        for layer in self.layers:
            layer.self_attn.is_causal = False

        self.rotary_emb = Qwen3RotaryEmbedding(cfg)
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, hidden_size] → [B, N, hidden_size]."""
        bsz, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.rotary_emb(x, position_ids)

        for layer in self.layers:
            x = layer(
                hidden_states=x,
                attention_mask=None,                 # bi-directional, no causal/pad mask
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )

        return self.norm(x)


class MetaQueryConnector(nn.Module):
    """Project N query hidden states from an MLLM into a diffusion-model
    cross-attention space, using a Qwen3-style bidirectional encoder.

    Args:
        in_dim:       MLLM hidden size (e.g. 2560 for Qwen3-VL-4B).
        out_dim:      target cross-attn dim (e.g. 4096 for Wan / UMT5-XXL).
        num_layers:   Qwen3Encoder depth.
        num_heads:    attention heads inside the encoder.
        num_kv_heads: GQA group count for the encoder (defaults to num_heads).
        ffn_mult:     FFN expansion factor inside the encoder.
        rope_theta:   RoPE base for the encoder.
        rms_init:     final RMSNorm initial scale (MetaQuery uses
                      sqrt(5.5)≈2.345 for Sana; Wan default is 1.0).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 8,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        ffn_mult: int = 4,
        rope_theta: float = 1_000_000.0,
        rms_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = Qwen3Encoder(
            hidden_size=in_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_dim=in_dim * ffn_mult,
            rope_theta=rope_theta,
        )
        self.proj1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU(approximate="tanh")
        self.proj2 = nn.Linear(out_dim, out_dim)
        self.norm = _RMSNorm(out_dim, eps=1e-5, init_scale=rms_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, in_dim] → [B, N, out_dim]."""
        x = self.encoder(x)
        x = self.proj2(self.act(self.proj1(x)))
        return self.norm(x)
