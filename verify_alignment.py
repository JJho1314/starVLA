"""FastWAM <-> starVLA Wan_FastWAM checkpoint key remapping.

Forward (FastWAM ckpt -> starVLA model state_dict):
  - `remap_fastwam_action_to_ours(mot)` :: maps `mixtures.action.*` keys to the
     keys consumed by `starVLA.model.modules.action_model.FastWAM_ActionDiT`.
  - `remap_fastwam_video_to_diffusers(mot)` :: maps `mixtures.video.*` keys to
     the keys consumed by `diffusers.WanTransformer3DModel`.

Both forward maps are 1:1 renames with no shape transforms — the lookup table
is stored next to this script as `verify_alignment_keymap.pt` (was derived
empirically from the released `libero_uncond_2cam224.pt` and verified to
strict-load). To regenerate, run `make_keymap()` against an official ckpt.

Reverse (starVLA state_dict -> FastWAM ckpt) is handled by
`convert_starvla_to_fastwam_ckpt.py`, which inverts these tables to roundtrip
trained starVLA weights through FastWAM's official eval pipeline.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch

_KEYMAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_alignment_keymap.pt")
_keymap_cache: Optional[dict] = None


def _load_keymap() -> dict:
    global _keymap_cache
    if _keymap_cache is None:
        if not os.path.exists(_KEYMAP_PATH):
            raise FileNotFoundError(
                f"verify_alignment_keymap.pt not found at {_KEYMAP_PATH}; "
                f"regenerate with make_keymap(<official_ckpt>)."
            )
        _keymap_cache = torch.load(_KEYMAP_PATH, weights_only=False)
    return _keymap_cache


def remap_fastwam_action_to_ours(mot: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """FastWAM mot dict (action subset) -> starVLA action_expert state_dict."""
    table = _load_keymap()["action"]  # fastwam_key -> starvla_key
    out: Dict[str, torch.Tensor] = {}
    for fk, sk in table.items():
        if fk in mot:
            out[sk] = mot[fk]
    return out


def remap_fastwam_video_to_diffusers(
    mot: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """[Pre-C path] FastWAM mot dict (video subset) -> diffusers WanTransformer3DModel state_dict.

    Kept for backward compat with old code paths (e.g. eq_phase2_starvla on a
    diffusers-backed model). After the C refactor the runtime model uses
    `WanVideoDiT` and you should call `remap_fastwam_video_to_wanvideo` instead.

    Returns (state_dict, unmapped_video_keys).
    """
    table = _load_keymap()["video"]
    out: Dict[str, torch.Tensor] = {}
    for fk, sk in table.items():
        if fk in mot:
            out[sk] = mot[fk]
    unmapped = [k for k in mot if k.startswith("mixtures.video.") and k not in table]
    return out, unmapped


def remap_fastwam_video_to_wanvideo(
    mot: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """FastWAM mot dict (video subset) -> WanVideoDiT state_dict (post-C refactor).

    starVLA's Wan2 backbone is now FastWAM's `WanVideoDiT` (vendored in
    `world_model.wan_video_dit`). Its key layout matches FastWAM's `mot.video.X`
    minus the `mixtures.video.` prefix, so this is a simple prefix-strip — no
    actual rename table needed. We still validate that every stripped key is one
    we expect, falling back to the keymap to catch surprise changes.

    Returns (state_dict, unmapped_video_keys).
    """
    out: Dict[str, torch.Tensor] = {}
    unmapped: List[str] = []
    PFX = "mixtures.video."
    for fk, tensor in mot.items():
        if not fk.startswith(PFX):
            continue
        wv_key = fk[len(PFX):]
        out[wv_key] = tensor
    # Optional sanity: every produced key should appear in the keymap (which lists
    # the canonical fastwam→diffusers mapping; the LHS is the same key set we expect
    # WanVideoDiT to want).
    table = _load_keymap()["video"]
    expected = {fk[len(PFX):] for fk in table.keys() if fk.startswith(PFX)}
    for wv_key in list(out.keys()):
        if wv_key not in expected:
            unmapped.append(wv_key)
            out.pop(wv_key)
    return out, unmapped


def map_video_key(fastwam_key: str) -> Optional[str]:
    """Single-key forward lookup for video DiT params."""
    return _load_keymap()["video"].get(fastwam_key)


def make_keymap(official_ckpt: str, out_path: str = _KEYMAP_PATH) -> None:
    """Regenerate the keymap by probing a held-out forward implementation.

    This requires the legacy bytecode-only `verify_alignment.cpython-310.pyc`
    (or any other ground-truth forward) to be importable as `_legacy`. Use
    only if the saved asset is lost — normal users should never need this.
    """
    import importlib.util
    legacy_pyc = os.path.join(os.path.dirname(__file__), "__pycache__", "verify_alignment.cpython-310.pyc")
    spec = importlib.util.spec_from_file_location("_legacy", legacy_pyc)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy forward from {legacy_pyc}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ck = torch.load(official_ckpt, map_location="cpu", weights_only=False)
    mot = ck["mot"]
    video_table: Dict[str, str] = {}
    action_table: Dict[str, str] = {}
    for fk, fv in mot.items():
        if fk.startswith("mixtures.video."):
            out, _ = mod.remap_fastwam_video_to_diffusers({fk: fv})
            for sk in out:
                video_table[fk] = sk
        elif fk.startswith("mixtures.action."):
            out = mod.remap_fastwam_action_to_ours({fk: fv})
            for sk in out:
                action_table[fk] = sk
    torch.save({"video": video_table, "action": action_table}, out_path)
    print(f"saved {out_path}: {len(video_table)} video + {len(action_table)} action entries")


def diff_state_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> dict:
    """Return missing/unexpected/shape-mismatch sets between two state_dicts."""
    ka, kb = set(a), set(b)
    return {
        "missing": sorted(ka - kb),
        "unexpected": sorted(kb - ka),
        "shape_mismatch": sorted(k for k in ka & kb if a[k].shape != b[k].shape),
    }


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_make = sub.add_parser("make-keymap", help="regenerate keymap from legacy .pyc")
    p_make.add_argument("--official-ckpt", required=True)
    p_make.add_argument("--out", default=_KEYMAP_PATH)
    p_check = sub.add_parser("check", help="report mot key counts in an official ckpt")
    p_check.add_argument("ckpt")
    args = p.parse_args()
    if args.cmd == "make-keymap":
        make_keymap(args.official_ckpt, args.out)
    elif args.cmd == "check":
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mot = ck["mot"]
        v = sum(1 for k in mot if k.startswith("mixtures.video."))
        a = sum(1 for k in mot if k.startswith("mixtures.action."))
        print(f"mot: {len(mot)} keys = {v} video + {a} action")
        act = remap_fastwam_action_to_ours(mot)
        vid, unm = remap_fastwam_video_to_diffusers(mot)
        print(f"forward: {len(act)} action + {len(vid)} video; {len(unm)} unmapped video")


if __name__ == "__main__":
    main()
