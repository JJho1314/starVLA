# Copyright 2025 starVLA community.
"""Joint MoT attention mask helpers — 1:1 reproduction of FastWAM's mask logic.

Two functions:
  - `build_video_to_video_mask`  — V→V block, mirrors
    FastWAM `WanVideoDiT.build_video_to_video_mask`
    (`fastwam/models/wan22/wan_video_dit.py:473`).
  - `build_mot_attention_mask`   — full [Sv+Sa, Sv+Sa] joint MoT mask, mirrors
    FastWAM `FastWAM._build_mot_attention_mask`
    (`fastwam/models/wan22/fastwam.py:386`).

Critical invariants (these are what make FastWAM's prefill-then-reuse inference
path mathematically equivalent to its training-time joint attention):

  V→A:                    all False        (video does not see action)
  A→V (first frame):      all True
  A→V (non-first frame):  all False
  A→A:                    all True         (FastWAM uncond default)

The three video-mask modes match FastWAM exactly:
  - `bidirectional`      : every video token sees every other
  - `per_frame_causal`   : tril over frames; within attended frames, full vis
  - `first_frame_causal` : every token sees everything EXCEPT first-frame query
                           rows are blocked from non-first-frame key columns
                           (so the first frame is causal-anchored)
"""

from __future__ import annotations

from typing import Union

import torch


def build_video_to_video_mask(
    video_seq_len: int,
    video_tokens_per_frame: int,
    mode: str = "first_frame_causal",
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build the V→V self-attention mask block. Returns bool tensor [Sv, Sv]."""
    if video_seq_len <= 0:
        raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
    if video_tokens_per_frame <= 0:
        raise ValueError(
            f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}"
        )

    if mode == "bidirectional":
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    if mode == "per_frame_causal":
        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                "`video_seq_len` must be divisible by `video_tokens_per_frame` in "
                f"`per_frame_causal` mode, got {video_seq_len} and {video_tokens_per_frame}"
            )
        num_video_frames = video_seq_len // video_tokens_per_frame
        frame_causal = torch.tril(
            torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
        )
        return frame_causal.repeat_interleave(
            video_tokens_per_frame, dim=0
        ).repeat_interleave(video_tokens_per_frame, dim=1)

    if mode == "first_frame_causal":
        video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        # First-frame query rows cannot see non-first-frame key columns.
        video_mask[:first_frame_tokens, first_frame_tokens:] = False
        return video_mask

    raise ValueError(f"Unsupported video attention mask mode: {mode}")


def build_mot_attention_mask(
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    video_attention_mask_mode: str = "first_frame_causal",
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Full joint MoT attention mask. Returns bool tensor [Sv+Sa, Sv+Sa].

    Layout::

                          key:  [...video Sv...] [...action Sa...]
                        ┌──────────────────────┬──────────────────────┐
        query video[Sv] │  V→V (mode-driven)   │  V→A: all False      │
                        ├──────────────────────┼──────────────────────┤
        query action[Sa]│  A→V: True only on   │  A→A: all True       │
                        │  first_frame_tokens  │                      │
                        └──────────────────────┴──────────────────────┘

    The V→A=False region is what makes inference KV-cache reuse safe: video
    tokens are never updated by action, so the K/V we cache from a single
    clean video prefill stay valid across all action denoising steps.
    """
    if video_seq_len <= 0:
        raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
    if action_seq_len <= 0:
        raise ValueError(f"`action_seq_len` must be positive, got {action_seq_len}")

    total_seq_len = video_seq_len + action_seq_len
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

    # V → V
    mask[:video_seq_len, :video_seq_len] = build_video_to_video_mask(
        video_seq_len=video_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        mode=video_attention_mask_mode,
        device=device,
    )
    # A → A
    mask[video_seq_len:, video_seq_len:] = True
    # A → V (first frame only)
    first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
    mask[video_seq_len:, :first_frame_tokens] = True
    # V → A stays all False from the zeros init.

    return mask
