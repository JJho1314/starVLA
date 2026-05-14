"""Loss helpers for CoVT-style mask supervision.

Implements dice/sigmoid focal loss and Hungarian matching between predicted
mask token outputs (``[N, H, W]``) and ground-truth masks (``[M, H, W]``).
References: LISA, MaskFormer, CoVT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss. ``inputs`` are raw logits, ``targets`` ∈ {0,1}."""
    p = inputs.sigmoid().flatten(1)
    t = targets.flatten(1)
    num = 2 * (p * t).sum(-1)
    den = p.sum(-1) + t.sum(-1)
    return (1 - (num + eps) / (den + eps)).mean()


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = inputs.sigmoid().clamp(eps, 1 - eps)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    modulator = (1 - p_t).pow(gamma)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * modulator * ce).flatten(1).mean(1).mean()


@torch.no_grad()
def _dice_cost(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise 1 - dice(pred[i], gt[j]) cost matrix.

    Args:
        pred: ``[N, H, W]`` logits.
        gt:   ``[M, H, W]`` binary GT.
    Returns:
        ``[N, M]`` cost.
    """
    p = pred.sigmoid().flatten(1)            # [N, H*W]
    g = gt.flatten(1).float()                # [M, H*W]
    inter = p @ g.T                          # [N, M]
    denom = p.sum(-1, keepdim=True) + g.sum(-1).unsqueeze(0)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice


def hungarian_match(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal 1-to-1 assignment via Hungarian algorithm.

    Returns:
        (row_idx, col_idx) tensors on ``pred.device``.
    """
    cost = _dice_cost(pred.float(), gt.float())
    r, c = linear_sum_assignment(cost.cpu().numpy())
    return (
        torch.as_tensor(r, device=pred.device),
        torch.as_tensor(c, device=pred.device),
    )


def matched_seg_loss(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    null_weight: float = 0.1,
) -> torch.Tensor:
    """CoVT-style per-image seg loss with Hungarian matching.

    - Match each GT mask to its best predicted mask.
    - Matched pairs: dice + BCE (focal).
    - Unmatched predictions: pushed to all-zero mask (BCE).

    Args:
        pred_masks: ``[N, H, W]`` logits (from learnable tokens).
        gt_masks:   ``[M, H, W]`` binary GT (M ≥ 0).
    """
    if gt_masks.numel() == 0 or gt_masks.shape[0] == 0:
        return F.binary_cross_entropy_with_logits(pred_masks, torch.zeros_like(pred_masks))

    r, c = hungarian_match(pred_masks, gt_masks)
    matched_pred = pred_masks[r]
    matched_gt = gt_masks[c].float()

    loss_pos = dice_loss(matched_pred, matched_gt) + sigmoid_focal_loss(matched_pred, matched_gt)

    unmatched = torch.tensor(
        [i for i in range(pred_masks.shape[0]) if i not in r.tolist()],
        device=pred_masks.device, dtype=torch.long,
    )
    if unmatched.numel() > 0:
        null_target = torch.zeros_like(pred_masks[unmatched])
        loss_neg = F.binary_cross_entropy_with_logits(pred_masks[unmatched], null_target)
        return loss_pos + null_weight * loss_neg
    return loss_pos
