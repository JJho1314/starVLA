# Copyright 2025 starVLA community. All rights reserved.
"""FastWAM-style Cosmos-Predict2 Video DiT (re-implementation for joint MoT).

Why this file exists:
  ``diffusers.CosmosTransformer3DModel`` has a monolithic ``forward()`` that
  doesn't expose per-block ``self_attn.q/k/v`` linears, RoPE freqs, or split
  ``pre_dit``/``post_dit`` methods. FastWAM's ``MoT`` joint-attention module
  requires all three (see ``starVLA/model/modules/world_model/mot.py``):

    1. ``pre_dit(x, t, ctx, ...)`` → dict with {tokens, freqs, t_mod, context,
       context_mask, meta}.
    2. ``blocks: nn.ModuleList`` where each block exposes
       ``block.self_attn.{q, k, v, o, norm_q, norm_k}``, ``block.cross_attn.*``,
       ``block.norm1/norm2/norm3``, ``block.modulation``, ``block.ffn``,
       ``block.num_heads``, ``block.attn_head_dim``.
    3. ``post_dit(tokens, pre_state)`` → output volume.

  This file re-implements Cosmos's DiT using the same module shapes
  ``fastwam.models.wan22.wan_video_dit.WanVideoDiT`` uses, so that MoT can
  treat ``CosmosVideoDiT`` and ``WanVideoDiT`` interchangeably as the
  ``video`` expert.

Weight loading:
  ``CosmosVideoDiT.load_from_diffusers_state_dict(state_dict)`` remaps
  diffusers' ``transformer_blocks.{i}.attn1.{to_q,to_k,to_v,to_out.0}`` →
  FastWAM-style ``blocks.{i}.self_attn.{q,k,v,o}``, etc. The remap is
  documented inline.

Caveats / divergences from diffusers Cosmos:
  - ``CosmosAdaLayerNormZero`` (per-block AdaLN-Lora with shift+gate from
    embedded_timestep + temb) vs. FastWAM's pre-modulated ``norm1/2`` with
    shift/scale/gate from ``t_mod``. We use FastWAM's modulation (a single
    learnable param chunk-6'd) plus a Wan-style per-token ``t_mod`` from
    ``time_projection``. This is mathematically NOT the same as diffusers'
    adaln-lora; loading diffusers AdaLN weights into this DiT requires a
    careful projection (TODO: see ``_remap_adaln_to_modulation``).
  - ``CosmosRotaryPosEmbed`` uses (cos, sin) pair freqs scaled by NTK; the
    FastWAM ``rope_apply`` consumes a flat ``freqs`` tensor that's the
    product of complex exponentials. We compute the equivalent Cosmos freqs
    in ``_precompute_cosmos_freqs`` to match the diffusers semantics.
  - ``CosmosLearnablePositionalEmbed`` is dropped here (Cosmos uses it for
    image-only pos signal at large training res 480×832 — for LIBERO 224×448
    it's not critical and would slow things down). Can be re-added later.

Status: SCAFFOLD. Module skeleton + state-dict remap stubs are present. The
remap function is the main work item for full numerical parity with diffusers.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Building blocks — mirror fastwam.models.wan22.wan_video_dit shapes
# ---------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Identical to FastWAM ``WanVideoDiT.RMSNorm`` so MoT can treat both
    backbones' norm layers as interchangeable."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        n = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).to(dtype) * self.weight


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Same formula as FastWAM ``sinusoidal_embedding_1d``.

    Always emits float32 — the downstream time_embedding Linear is bf16/fp32,
    so casting back to ``position.dtype`` (which can be Long when called from
    int timesteps) would crash F.linear.
    """
    half_dim = dim // 2
    position = position.float()
    sinusoid = position[:, None] * torch.exp(
        torch.arange(half_dim, device=position.device, dtype=torch.float32)
        * -(math.log(10000.0) / (half_dim - 1))
    )
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)


def _rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply RoPE — IDENTICAL math to FastWAM ``wan_video_dit.rope_apply``.

    Args:
        x: ``[B, S, n*D]`` tokens.
        freqs: complex tensor ``[S, 1, D/2]`` from
            ``_precompute_cosmos_freqs``. (Same convention as FastWAM's
            ``precompute_freqs_cis``: ``torch.polar(ones, theta)``.)
        num_heads: n.

    MoT calls FastWAM's ``rope_apply`` directly via the block's ``self_attn``;
    we mirror the same complex-multiply math here so our standalone
    ``CosmosDiTBlock.forward`` produces the same outputs as the MoT path.
    """
    B, S, HD = x.shape
    D = HD // num_heads
    x = x.view(B, S, num_heads, D)
    # Treat each (even, odd) pair of head_dim as a complex number.
    x_cplx = torch.view_as_complex(
        x.to(torch.float64).reshape(B, S, num_heads, D // 2, 2)
    )
    # Multiply by complex freqs (broadcast over batch + heads).
    x_rot = torch.view_as_real(x_cplx * freqs).flatten(3)  # [B, S, n, D]
    return x_rot.reshape(B, S, HD).to(x.dtype)


class _SelfAttention(nn.Module):
    """Same shape as FastWAM ``SelfAttention`` (q/k/v/o + norm_q/norm_k).

    The split into separate q/k/v linears is critical for MoT — it lets the
    joint module call ``block.self_attn.q/k/v`` to extract Q/K/V before the
    SDPA, then run the SDPA across video + action together.
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        H = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, H, bias=False)   # Cosmos: out_bias=False
        self.k = nn.Linear(hidden_dim, H, bias=False)
        self.v = nn.Linear(hidden_dim, H, bias=False)
        self.o = nn.Linear(H, hidden_dim, bias=False)
        self.norm_q = RMSNorm(H, eps=eps)
        self.norm_k = RMSNorm(H, eps=eps)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = _rope_apply(q, freqs, self.num_heads)
        k = _rope_apply(k, freqs, self.num_heads)
        # Standard SDPA. MoT bypasses this method entirely — it builds Q/K/V
        # directly from .q/.k/.v and runs a joint SDPA over video+action.
        B, S, HD = q.shape
        D = self.attn_head_dim
        q = q.view(B, S, self.num_heads, D).transpose(1, 2)
        k = k.view(B, S, self.num_heads, D).transpose(1, 2)
        v = v.view(B, S, self.num_heads, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self_attn_mask)
        out = out.transpose(1, 2).reshape(B, S, HD)
        return self.o(out)


class _CrossAttention(nn.Module):
    """Same shape as FastWAM ``CrossAttention``."""

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        H = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, H, bias=False)
        self.k = nn.Linear(hidden_dim, H, bias=False)
        self.v = nn.Linear(hidden_dim, H, bias=False)
        self.o = nn.Linear(H, hidden_dim, bias=False)
        self.norm_q = RMSNorm(H, eps=eps)
        self.norm_k = RMSNorm(H, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        B, S, HD = q.shape
        D = self.attn_head_dim
        q = q.view(B, S, self.num_heads, D).transpose(1, 2)
        k = k.view(B, ctx.shape[1], self.num_heads, D).transpose(1, 2)
        v = v.view(B, ctx.shape[1], self.num_heads, D).transpose(1, 2)
        attn_mask = None
        if ctx_mask is not None:
            # MoT and FastWAM block can pass ctx_mask as 2D/3D/4D. SDPA wants
            # broadcastable to ``[B, num_heads, S_q, S_k]``.
            if ctx_mask.dim() == 2:        # [B, L]
                attn_mask = ctx_mask[:, None, None, :].expand(B, 1, S, ctx_mask.shape[-1])
            elif ctx_mask.dim() == 3:      # [B, S, L]
                attn_mask = ctx_mask.unsqueeze(1)              # [B, 1, S, L]
            elif ctx_mask.dim() == 4:      # already broadcastable [B, 1, S, L]
                attn_mask = ctx_mask
            else:
                raise ValueError(f"ctx_mask must be 2D/3D/4D, got {ctx_mask.dim()}D")
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, S, HD)
        return self.o(out)


class _GateModule(nn.Module):
    """Mirror FastWAM ``DiTBlock.gate``. MoT calls ``block.gate(x, gate, residual)``
    directly when stitching the joint-attention output back into each expert,
    so the block must expose this submodule under the exact name."""

    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual


class CosmosDiTBlock(nn.Module):
    """A Cosmos transformer block in FastWAM-block layout.

    Identical attribute names to ``fastwam.models.wan22.wan_video_dit.DiTBlock``
    so MoT's ``compute_qkv`` / ``rope_apply`` codepaths Just Work. Forward
    matches Wan's per-block algorithm:

        x = x + gate_msa * self_attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + cross_attn(norm3(x), context, ctx_mask)
        x = x + gate_mlp * ffn(modulate(norm2(x), shift_mlp, scale_mlp))

    The Cosmos diffusers block instead uses AdaLN-Lora gating (norm1/2/3 each
    return their own gate tensor from embedded_timestep + temb). Numerically
    different but mathematically equivalent in expressivity. The weight remap
    in ``CosmosVideoDiT.load_from_diffusers_state_dict`` collapses Cosmos's
    AdaLN-Lora into FastWAM's per-block modulation parameter.
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int,
                 ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = _SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = _CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        # FFN: bias=False to match Cosmos (out_bias=False).
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )
        # 6-way modulation param (shift_msa, scale_msa, gate_msa,
        #                        shift_mlp, scale_mlp, gate_mlp).
        # Matches FastWAM DiTBlock.modulation exactly.
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim ** 0.5)
        self.gate = _GateModule()

    def forward(self, x: torch.Tensor, context: torch.Tensor, t_mod: torch.Tensor,
                freqs: torch.Tensor, context_mask: Optional[torch.Tensor] = None,
                self_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Same control-flow as FastWAM DiTBlock.forward.
        has_seq = (t_mod.dim() == 4)
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(device=t_mod.device, dtype=t_mod.dtype) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        def _modulate(h, s, sc):
            return h * (1 + sc) + s

        attn_in = _modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(attn_in, freqs, self_attn_mask=self_attn_mask)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        ffn_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(ffn_in)
        return x


class _Head(nn.Module):
    """FastWAM-style head: norm → modulated linear to (out_dim × prod(patch))."""

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size), bias=False)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if t_mod.dim() == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(device=t_mod.device, dtype=t_mod.dtype)
                            + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(device=t_mod.device, dtype=t_mod.dtype) + t_mod).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


# ---------------------------------------------------------------------
# Cosmos RoPE (NTK-scaled 3D) → FastWAM flat ``freqs`` tensor
# ---------------------------------------------------------------------
def _precompute_cosmos_freqs(
    attn_head_dim: int,
    max_size: Tuple[int, int, int] = (128, 240, 240),
    rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
    base_fps: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate ``CosmosRotaryPosEmbed`` math but produce per-axis 1D freqs.

    Returns:
        (freqs_t, freqs_h, freqs_w) — each shape ``[max_size_axis, D_axis]``
        containing the complex exponential rates. The caller (``pre_dit``)
        slices these to the current grid size and concatenates into the
        ``[S, 1, D]`` form ``_rope_apply`` expects.

    Note: Cosmos partitions the head_dim into (T, H, W) thirds with shape
        D_h = D_w = (attn_head_dim // 6) * 2
        D_t = attn_head_dim - D_h - D_w
    """
    D_h = (attn_head_dim // 6) * 2
    D_w = (attn_head_dim // 6) * 2
    D_t = attn_head_dim - D_h - D_w
    h_ntk = rope_scale[1] ** (D_h / max(1, D_h - 2))
    w_ntk = rope_scale[2] ** (D_w / max(1, D_w - 2))
    t_ntk = rope_scale[0] ** (D_t / max(1, D_t - 2))
    h_theta, w_theta, t_theta = 10000.0 * h_ntk, 10000.0 * w_ntk, 10000.0 * t_ntk

    def _axis(N: int, D: int, theta: float) -> torch.Tensor:
        # Match FastWAM's ``precompute_freqs_cis``: returns COMPLEX tensor
        # ``[N, D/2]`` via ``torch.polar(ones, theta_grid)``. MoT's
        # ``rope_apply`` (which our block also calls) consumes this exact form.
        if D <= 0:
            return torch.zeros(N, 0, dtype=torch.complex64)
        seq = torch.arange(N, dtype=torch.float64)
        dim_range = torch.arange(0, D, 2, dtype=torch.float64)[: D // 2] / D
        freqs_inv = 1.0 / (theta ** dim_range)
        freqs = torch.outer(seq, freqs_inv)                    # [N, D/2] real
        return torch.polar(torch.ones_like(freqs), freqs)      # [N, D/2] complex

    return (
        _axis(max_size[0], D_t, t_theta),
        _axis(max_size[1], D_h, h_theta),
        _axis(max_size[2], D_w, w_theta),
    )


# ---------------------------------------------------------------------
# Main DiT
# ---------------------------------------------------------------------

class CosmosVideoDiT(nn.Module):
    """Cosmos-Predict2 DiT in FastWAM-block layout for MoT compatibility.

    Public surface mirrors ``fastwam.models.wan22.wan_video_dit.WanVideoDiT``
    so the WanFastWAM framework can swap backbones with a single dispatcher
    flag. The class exposes ``pre_dit``, ``post_dit``, and a ``blocks``
    ``ModuleList`` whose elements are ``CosmosDiTBlock`` instances.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        in_dim: int = 17,          # Cosmos: 16 latent + 1 condition_mask
        ffn_dim: int = 8192,        # mlp_ratio=4 × 2048 hidden
        out_dim: int = 16,
        text_dim: int = 1024,       # Cosmos T5XXL
        freq_dim: int = 256,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 16,
        attn_head_dim: int = 128,
        num_layers: int = 28,
        seperated_timestep: bool = True,
        fuse_vae_embedding_in_latents: bool = True,
        # Cosmos-specific kwargs (not used by Wan but accepted for compat)
        max_size: Tuple[int, int, int] = (128, 240, 240),
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        use_gradient_checkpointing: bool = False,
        # ignored (Wan-only) — accepted to share the same yaml schema
        has_image_input: bool = False,
        require_clip_embedding: bool = False,
        require_vae_embedding: bool = False,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode: str = "group_diagonal",
        video_attention_mask_mode: str = "first_frame_causal",
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        **_ignore,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim

        if action_conditioned:
            raise NotImplementedError(
                "CosmosVideoDiT doesn't yet support standalone action_conditioned mode. "
                "For action conditioning, use the MoT framework path with "
                "WanFastWAM joint training instead."
            )

        # 1) Patch embedding: Cosmos uses Linear-style patchify. We mirror Wan
        # with a Conv3d for parity — the diffusers weight remap below converts.
        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2) Text embedding (project T5 1024 → hidden_dim). Cosmos's cross-attn
        # accepts raw T5 (1024d) directly; FastWAM projects upfront so the
        # cross-attn in each block sees `hidden_dim` keys. We choose FastWAM's
        # convention so block code is uniform.
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3) Time embedding + projection (per-token, since fuse_vae_embedding_in_latents)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

        # 4) Transformer blocks
        self.blocks = nn.ModuleList([
            CosmosDiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        # 5) Head (norm + modulated linear)
        self.head = _Head(hidden_dim, out_dim, patch_size, eps)

        # 6) RoPE — store the 3 axis tables; sliced per-call in pre_dit.
        ft, fh, fw = _precompute_cosmos_freqs(
            attn_head_dim, max_size=max_size, rope_scale=rope_scale
        )
        # Concatenate cos/sin into the [S, 1, D] form _rope_apply expects.
        # FastWAM's rope_apply expects freqs already in (cos, sin) packed form
        # at axis-product time; we precompute the raw axis tables here and
        # do the product in pre_dit.
        self.register_buffer("_freqs_t", ft, persistent=False)
        self.register_buffer("_freqs_h", fh, persistent=False)
        self.register_buffer("_freqs_w", fw, persistent=False)

        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        logger.info(
            f"[CosmosVideoDiT] layers={num_layers} hidden={hidden_dim} "
            f"heads={num_heads}×{attn_head_dim}={num_heads * attn_head_dim} "
            f"text_dim={text_dim} patch_size={patch_size} in_dim={in_dim}"
        )

    # ----------------------------------------------------------
    # patchify / unpatchify (same as Wan)
    # ----------------------------------------------------------
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        return rearrange(
            x, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )

    # ----------------------------------------------------------
    # pre_dit / post_dit — identical contract to WanVideoDiT
    # ----------------------------------------------------------
    def _build_freqs(self, f: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Slice precomputed COMPLEX axis freqs and concat → ``[f*h*w, 1, D/2]``.

        Same form as ``WanVideoDiT.pre_dit``'s freqs (complex tensor, half head
        dim). MoT's ``rope_apply`` (and our local ``_rope_apply``) does
        ``view_as_complex(x_pairs) * freqs``, so freqs must be complex of length
        ``D/2``.
        """
        ft = self._freqs_t[:f]  # [f, D_t/2] complex
        fh = self._freqs_h[:h]  # [h, D_h/2] complex
        fw = self._freqs_w[:w]  # [w, D_w/2] complex
        # Broadcast each axis to grid shape, then concat along last dim.
        ft_b = ft.view(f, 1, 1, -1).expand(f, h, w, -1)
        fh_b = fh.view(1, h, 1, -1).expand(f, h, w, -1)
        fw_b = fw.view(1, 1, w, -1).expand(f, h, w, -1)
        freqs = torch.cat([ft_b, fh_b, fw_b], dim=-1)  # [f,h,w, D/2] complex
        return freqs.reshape(f * h * w, 1, -1).to(device)

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Same control-flow shape as WanVideoDiT.pre_dit.
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B,C,T,H,W], got {tuple(x.shape)}")
        batch_size = x.shape[0]
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got {tuple(timestep.shape)}")
        if context_mask is None:
            context_mask = torch.ones(
                (context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device
            )

        # ----- Cosmos input channels: append condition_mask (+ optional
        # padding_mask) to bring channels from latent_z_dim to in_dim ----
        # WanFastWAM framework passes pure z-space latents [B, z_dim, T, H, W].
        # Cosmos DiT expects [B, z_dim + 1 (+ 1), T, H, W] with:
        #   - condition_mask  : 1.0 on frame 0 (TI2V anchor), 0.0 elsewhere
        #   - padding_mask    : all 0 (no padding) — concat only if needed
        z_dim_in = x.shape[1]
        T_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
        if z_dim_in < self.in_dim:
            extras = []
            # condition_mask: anchor frame 0
            cond = x.new_zeros(batch_size, 1, T_in, H_in, W_in)
            cond[:, :, 0:1] = 1.0
            extras.append(cond)
            if z_dim_in + 1 < self.in_dim:
                # padding_mask: all zeros (no padding)
                pad = x.new_zeros(batch_size, 1, T_in, H_in, W_in)
                extras.append(pad)
            x = torch.cat([x] + extras, dim=1)
            if x.shape[1] != self.in_dim:
                raise ValueError(
                    f"Channel-pad to in_dim={self.in_dim} failed: got {x.shape[1]} channels. "
                    f"Caller passed z_dim={z_dim_in}, expected z_dim + condition_mask "
                    f"(+ padding_mask if concat_padding_mask=True) to total in_dim."
                )

        patch_h, patch_w = int(self.patch_size[1]), int(self.patch_size[2])
        if x.shape[3] % patch_h or x.shape[4] % patch_w:
            raise ValueError(
                f"Latent HxW=({x.shape[3]},{x.shape[4]}) must be divisible by "
                f"patch ({patch_h},{patch_w})."
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        # Per-token time embedding (first frame anchored at timestep=0 to match
        # FastWAM TI2V conditioning).
        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            tok_t = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype, device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            tok_t[:, 0, :] = 0
            tok_t = tok_t.reshape(batch_size, -1)
            # _sinusoidal_embedding_1d always returns fp32 for numerical accuracy.
            # When pre_dit is called OUTSIDE an autocast context (e.g. from
            # ``WanFastWAM._prefill_video_cache`` which doesn't wrap pre_dit in
            # autocast, only the subsequent MoT call), the bf16 Linear weights
            # would mismatch. Cast t_emb to the time_embedding weight dtype.
            t_emb = _sinusoidal_embedding_1d(self.freq_dim, tok_t.reshape(-1))
            t_emb = t_emb.to(dtype=self.time_embedding[0].weight.dtype)
            t = self.time_embedding(t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError(
                "CosmosVideoDiT only supports seperated_timestep + "
                "fuse_vae_embedding_in_latents for now (same as Wan)."
            )

        x = self.patchify(x)
        f, h, w = x.shape[2:]
        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        # Text projection 1024 → hidden_dim (same as Wan's text_embedding).
        context = self.text_embedding(context)
        # Expand context_mask to [B, S, L] so cross-attn can use it directly.
        context_mask_expanded = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)

        freqs = self._build_freqs(f, h, w, x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask_expanded,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        return self.unpatchify(x, (f, h, w))

    # ----------------------------------------------------------
    # Standalone forward (when not routed through MoT) — same as Wan
    # ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ) -> torch.Tensor:
        pre = self.pre_dit(
            x=x, timestep=timestep, context=context,
            context_mask=context_mask, action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        toks = pre["tokens"]
        for block in self.blocks:
            toks = block(
                x=toks, context=pre["context"], t_mod=pre["t_mod"],
                freqs=pre["freqs"], context_mask=pre["context_mask"],
            )
        return self.post_dit(toks, pre)

    # ----------------------------------------------------------
    # Weight loading — diffusers state_dict → this layout
    # ----------------------------------------------------------
    def load_from_diffusers_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> Dict[str, list]:
        """Remap diffusers ``CosmosTransformer3DModel`` keys → this DiT.

        Returns ``{"missing": [...], "unexpected": [...]}`` so caller can audit.

        Status: STUB. Maps the easy bits (patch_embed, blocks.attn.{q,k,v,o},
        blocks.ff, head). The hard bit — ``CosmosAdaLayerNormZero.embedding_adaln_lora`` →
        FastWAM ``modulation`` parameter — is *not* implemented and the
        corresponding entries are reported as missing. To finish that part,
        either (a) re-derive equivalent per-block shift/scale/gate from the
        diffusers AdaLN-Lora MLPs (needs running both backbones on a probe
        input and least-squares fitting), or (b) abandon weight loading and
        train CosmosVideoDiT from scratch (acceptable since Cosmos pretrained
        on Video2World, not action). For the LIBERO experiments here, (b) is
        the simpler path — joint MoT training will learn the modulation
        params during the 21700-step run.
        """
        loaded = []
        missing = []
        own = self.state_dict()

        def _try_copy(dst_key: str, src_key: str):
            if src_key in state_dict and dst_key in own:
                src = state_dict[src_key]
                if src.shape == own[dst_key].shape:
                    own[dst_key].copy_(src)
                    loaded.append(dst_key)
                    return True
            missing.append(dst_key)
            return False

        # patch_embedding: diffusers uses Linear, we use Conv3d. Reshape weight.
        # diffusers key: "patch_embed.proj.weight" of shape [hidden, in*p_t*p_h*p_w]
        # our: patch_embedding.weight of shape [hidden, in, p_t, p_h, p_w]
        diff_key = "patch_embed.proj.weight"
        if diff_key in state_dict:
            w = state_dict[diff_key]
            try:
                w = w.reshape(self.hidden_dim, self.in_dim,
                              self.patch_size[0], self.patch_size[1], self.patch_size[2])
                own["patch_embedding.weight"].copy_(w)
                loaded.append("patch_embedding.weight")
            except Exception as e:
                logger.warning(f"patch_embed remap failed: {e}")
                missing.append("patch_embedding.weight")
        else:
            missing.append("patch_embedding.weight")

        # Per-block: attn1 → self_attn, attn2 → cross_attn, ff → ffn
        for i in range(len(self.blocks)):
            d = f"transformer_blocks.{i}"
            o = f"blocks.{i}"
            # self_attn (attn1)
            for src_n, dst_n in [("to_q", "q"), ("to_k", "k"), ("to_v", "v"),
                                  ("to_out.0", "o"), ("norm_q", "norm_q"), ("norm_k", "norm_k")]:
                if src_n.startswith("norm_"):
                    _try_copy(f"{o}.self_attn.{dst_n}.weight", f"{d}.attn1.{src_n}.weight")
                else:
                    _try_copy(f"{o}.self_attn.{dst_n}.weight", f"{d}.attn1.{src_n}.weight")
            # cross_attn (attn2)
            for src_n, dst_n in [("to_q", "q"), ("to_k", "k"), ("to_v", "v"),
                                  ("to_out.0", "o"), ("norm_q", "norm_q"), ("norm_k", "norm_k")]:
                _try_copy(f"{o}.cross_attn.{dst_n}.weight", f"{d}.attn2.{src_n}.weight")
            # FFN: diffusers FeedForward is GELU(Linear) → Linear with .net.0.proj.weight and .net.2.weight
            _try_copy(f"{o}.ffn.0.weight", f"{d}.ff.net.0.proj.weight")
            _try_copy(f"{o}.ffn.2.weight", f"{d}.ff.net.2.weight")
            # NOTE: modulation param NOT loaded — see docstring above.
            missing.append(f"{o}.modulation  [diffusers AdaLN-Lora → FastWAM modulation: NOT remapped]")

        # Head: norm_out + proj_out
        _try_copy("head.head.weight", "proj_out.weight")
        missing.append("head.modulation  [diffusers AdaLN → FastWAM modulation: NOT remapped]")

        result = {"loaded": loaded, "missing": missing}
        logger.info(
            f"[CosmosVideoDiT] load_from_diffusers: copied={len(loaded)} "
            f"missing={len(missing)} (modulation params NOT remapped — see docstring)"
        )
        if strict and missing:
            raise RuntimeError(
                f"Strict load failed; {len(missing)} keys missing/unmapped. "
                "Set strict=False to ignore (training will overwrite missing modulation)."
            )
        return result


__all__ = ["CosmosVideoDiT", "CosmosDiTBlock"]
