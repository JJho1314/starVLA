"""Pre-compute ActionDiT backbone warm-start from Cosmos via Wan-style linear
interpolation + alpha=sqrt(d_src/d_tgt) variance-preserving scaling.

Mirrors the FastWAM-official ``preprocess_action_dit_backbone.py`` recipe:
  1. F.interpolate(mode='linear', align_corners=True) on each mismatched dim
     (sequential 1D — same as the upstream script).
  2. alpha = sqrt(d_video / d_action) applied when the last dim is resized,
     to preserve activation variance after the projection-width change.

Unlike ``precompute_action_dit_backbone_cosmos.py`` (which assumed
hidden_dim parity Cosmos↔ActionDiT and did a direct copy), this script
targets the smaller ActionDiT (hidden=1024) layout that mirrors Wan's
ratio: residual 1024 with attention sub-space upcast to 2048 (16×128 =
Cosmos VideoDiT attn dim).

Use:
    python scripts/precompute_action_dit_backbone_cosmos_interp.py \\
        --cosmos_path /path/to/Cosmos-Predict2-2B-Video2World \\
        --out checkpoints/ActionDiT_cosmos_backbone_1024hdim.pt \\
        --target_hidden_dim 1024 --target_ffn_dim 4096
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _parse_bool(name: str) -> bool:
    return str(name).strip().lower() in {"1", "true", "yes", "y"}


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src
    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()
    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_hidden_dim", type=int, default=1024)
    ap.add_argument("--target_ffn_dim", type=int, default=4096)
    ap.add_argument("--target_action_dim", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--apply_alpha_scaling", default="true")
    args = ap.parse_args()

    apply_alpha = _parse_bool(args.apply_alpha_scaling)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # 1. Load diffusers Cosmos transformer config + state dict.
    print(f"[1/4] Loading diffusers Cosmos-Predict2 transformer from {args.cosmos_path} ...")
    from diffusers import CosmosTransformer3DModel
    diffusers_dit = CosmosTransformer3DModel.from_pretrained(
        args.cosmos_path, subfolder="transformer", torch_dtype=torch_dtype,
    )
    diffusers_state = diffusers_dit.state_dict()
    cfg = diffusers_dit.config
    src_hidden = cfg.num_attention_heads * cfg.attention_head_dim
    print(f"    Cosmos src: hidden={src_hidden}, heads={cfg.num_attention_heads}, "
          f"head_dim={cfg.attention_head_dim}, layers={cfg.num_layers}, text_dim={cfg.text_embed_dim}")
    del diffusers_dit

    # 2. Materialize CosmosVideoDiT with diffusers weights (so backbone state keys match ActionDiT's
    #    FastWAM-block layout, NOT the diffusers naming).
    print("[2/4] Building CosmosVideoDiT (FastWAM-block layout) and loading diffusers weights ...")
    from starVLA.model.modules.world_model.cosmos_video_dit import CosmosVideoDiT
    src_ffn = int(src_hidden * float(getattr(cfg, "mlp_ratio", 4.0)))
    src_in_dim = cfg.in_channels + (1 if getattr(cfg, "concat_padding_mask", True) else 0)
    cosmos_dit = CosmosVideoDiT(
        hidden_dim=src_hidden, in_dim=src_in_dim, ffn_dim=src_ffn,
        out_dim=cfg.out_channels, text_dim=cfg.text_embed_dim,
        patch_size=tuple(cfg.patch_size), num_heads=cfg.num_attention_heads,
        attn_head_dim=cfg.attention_head_dim, num_layers=cfg.num_layers,
        max_size=tuple(getattr(cfg, "max_size", (128, 240, 240))),
        rope_scale=tuple(getattr(cfg, "rope_scale", (2.0, 1.0, 1.0))),
    ).to(torch_dtype)
    rep = cosmos_dit.load_from_diffusers_state_dict(diffusers_state, strict=False)
    print(f"    diffusers→CosmosVideoDiT: loaded={len(rep['loaded'])} missing={len(rep['missing'])}")

    # 3. Build the target ActionDiT (smaller residual, same attn sub-space).
    print(f"[3/4] Building target ActionDiT with hidden_dim={args.target_hidden_dim} "
          f"ffn_dim={args.target_ffn_dim} (attn_hidden={cfg.num_attention_heads * cfg.attention_head_dim})")
    from starVLA.model.modules.action_model.FastWAM_ActionDiT import ActionDiT
    action_dit = ActionDiT(
        hidden_dim=args.target_hidden_dim,
        ffn_dim=args.target_ffn_dim,
        num_heads=cfg.num_attention_heads,
        attn_head_dim=cfg.attention_head_dim,
        num_layers=cfg.num_layers,
        text_dim=cfg.text_embed_dim,
        freq_dim=256,
        eps=1e-6,
        action_dim=args.target_action_dim,
    ).to(torch_dtype)

    # 4. For each ActionDiT backbone key, find equivalent CosmosVideoDiT key and resize.
    print(f"[4/4] Resizing Cosmos {src_hidden} → ActionDiT {args.target_hidden_dim} "
          f"(alpha_scaling={apply_alpha}) ...")
    action_state = action_dit.state_dict()
    cosmos_state = cosmos_dit.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    missing: list[str] = []  # keys with no Cosmos equivalent — fall back to ActionDiT init
    for k in sorted(backbone_keys):
        if k not in cosmos_state:
            # Keep ActionDiT's random init for these (typically biases when Cosmos
            # uses bias=False, plus modulation params that don't map cleanly).
            # MUST still be present in payload — ActionDiT's loader requires every
            # expected backbone key to exist (strict).
            backbone_state[k] = action_state[k].detach().to(device="cpu").contiguous()
            missing.append(k)
            continue
        src = cosmos_state[k]
        tgt = action_state[k]
        if tuple(src.shape) == tuple(tgt.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(tgt.shape))
            if apply_alpha and src.ndim >= 2 and src.shape[-1] != tgt.shape[-1]:
                alpha = (float(src.shape[-1]) / float(tgt.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state[k] = value.detach().to(dtype=tgt.dtype, device="cpu").contiguous()

    print(f"    copied={copied} interpolated={interpolated} "
          f"random_kept_from_actiondit_init={len(missing)} total={len(backbone_state)}")
    if missing:
        print(f"    first 5 random-kept keys: {missing[:5]}")

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state,
        "meta": {
            "hidden_dim": int(args.target_hidden_dim),
            "ffn_dim": int(args.target_ffn_dim),
            "num_layers": int(cfg.num_layers),
            "num_heads": int(cfg.num_attention_heads),
            "attn_head_dim": int(cfg.attention_head_dim),
            "text_dim": int(cfg.text_embed_dim),
            "freq_dim": 256,
            "eps": 1e-6,
            "source": "cosmos-predict2-2b-video2world",
            "src_hidden": int(src_hidden),
            "interp_method": "Wan-style sequential 1D linear interp + alpha=sqrt(d_src/d_tgt)",
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"    Saved {len(backbone_state)} keys to {out_path} ({size_mb:.1f} MB)")
    print(f"    Set framework.action_dit.pretrained_path: {out_path}")
    print(f"        framework.action_dit.skip_pretrained_load: false")


if __name__ == "__main__":
    main()
