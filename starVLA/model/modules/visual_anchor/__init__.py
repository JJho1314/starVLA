"""Visual anchor teachers for Chain-of-Visual-Thought (CoVT) style training.

Currently only ``sam`` is supported; depth/dino/edge can be added later by
following the same wrapper pattern.
"""

from .sam_anchor import SamAnchor

__all__ = ["SamAnchor", "build_anchors"]


def build_anchors(anchor_cfg: dict):
    """Instantiate the requested anchor teachers.

    Args:
        anchor_cfg: ``{"sam": {"checkpoint": "...", "model_type": "vit_h"}, ...}``.

    Returns:
        dict[str, nn.Module]: name -> teacher module (eval-mode, frozen).
    """
    anchors = {}
    if "sam" in anchor_cfg:
        anchors["sam"] = SamAnchor(**anchor_cfg["sam"])
    return anchors
