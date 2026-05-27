"""Pre-compute ActionDiT backbone warm-start from Cosmos-Predict2 DiT weights.

Sibling of FastWAM-official ``preprocess_action_dit_backbone.py`` (which does
``hidden_dim`` interpolation Wan22-3072 → ActionDiT-1024 with alpha scaling).
For Cosmos the dims already match (both 16 heads × 128 head_dim = 2048,
28 layers), so this script is a **direct block-weight copy** — no
interpolation needed.

What it does:
    1. Load CosmosVideoDiT and copy diffusers state dict into it via the
       existing remap (attn / ffn / head linears load; modulation params
       stay random — see CosmosVideoDiT.load_from_diffusers_state_dict).
    2. Extract a "backbone-only" state dict — keys matching ActionDiT's
       trainable backbone (blocks + time_embedding + time_projection +
       text_embedding). Skip ``action_encoder`` / ``head`` (action-specific).
    3. Save as a ``.pt`` payload with metadata, in the format ActionDiT
       loads via ``pretrained_path`` in the yaml.

Caveats:
    - Modulation params are not copied (Cosmos AdaLN-Lora can't be cleanly
      remapped to FastWAM modulation parameter). ActionDiT keeps its random
      modulation init for those entries.
    - Cosmos was video-pretrained, NOT action-pretrained — this is a
      transfer-learning warm-start, not a fully aligned init. The FastWAM
      paper's ~70% head-start claim was for Wan22 (which is action-aware
      via its TI2V conditioning); Cosmos warm-start probably gives less.

Use:
    python scripts/precompute_action_dit_backbone_cosmos.py \\
        --cosmos_path /data/.../Cosmos-Predict2-2B-Video2World \\
        --out checkpoints/ActionDiT_cosmos_backbone_2048hdim.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos_path", required=True,
                    help="Path to Cosmos-Predict2-2B-Video2World folder.")
    ap.add_argument("--out", required=True, help="Output .pt path for ActionDiT warm-start payload.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # Step 1: load Cosmos DiT via diffusers, get its state dict.
    print(f"[1/3] Loading diffusers Cosmos-Predict2 transformer from {args.cosmos_path} ...")
    from diffusers import CosmosTransformer3DModel
    diffusers_dit = CosmosTransformer3DModel.from_pretrained(
        args.cosmos_path, subfolder="transformer", torch_dtype=torch_dtype,
    )
    diffusers_state = diffusers_dit.state_dict()
    cfg = diffusers_dit.config
    print(f"    Cosmos cfg: layers={cfg.num_layers}, heads={cfg.num_attention_heads}, "
          f"head_dim={cfg.attention_head_dim} → hidden={cfg.num_attention_heads * cfg.attention_head_dim}")
    del diffusers_dit

    # Step 2: instantiate CosmosVideoDiT and copy weights.
    print(f"[2/3] Remapping diffusers state dict → CosmosVideoDiT (FastWAM-block layout) ...")
    from starVLA.model.modules.world_model.cosmos_video_dit import CosmosVideoDiT
    hidden = cfg.num_attention_heads * cfg.attention_head_dim
    ffn_dim = int(hidden * float(getattr(cfg, "mlp_ratio", 4.0)))
    in_dim = cfg.in_channels + (1 if getattr(cfg, "concat_padding_mask", True) else 0)
    dit = CosmosVideoDiT(
        hidden_dim=hidden, in_dim=in_dim, ffn_dim=ffn_dim, out_dim=cfg.out_channels,
        text_dim=cfg.text_embed_dim, patch_size=tuple(cfg.patch_size),
        num_heads=cfg.num_attention_heads, attn_head_dim=cfg.attention_head_dim,
        num_layers=cfg.num_layers,
        max_size=tuple(getattr(cfg, "max_size", (128, 240, 240))),
        rope_scale=tuple(getattr(cfg, "rope_scale", (2.0, 1.0, 1.0))),
    ).to(torch_dtype)
    report = dit.load_from_diffusers_state_dict(diffusers_state, strict=False)
    print(f"    remap report: loaded={len(report['loaded'])} missing={len(report['missing'])}")

    # Step 3: extract ActionDiT-shape backbone state dict.
    # ActionDiT structure (FastWAM_ActionDiT.ActionDiT.__init__):
    #   - action_encoder: Linear(action_dim, hidden_dim)   ← ACTION-SPECIFIC, SKIP
    #   - text_embedding: Sequential(Linear → GELU → Linear)
    #   - time_embedding: Sequential(Linear → SiLU → Linear)
    #   - time_projection: Sequential(SiLU → Linear)
    #   - blocks: ModuleList[DiTBlock(...)]
    #   - head: Linear(hidden_dim, action_dim)            ← ACTION-SPECIFIC, SKIP
    #
    # We map CosmosVideoDiT.{text_embedding, time_embedding, time_projection, blocks}
    # 1:1 onto ActionDiT (same module structure).
    print(f"[3/3] Extracting ActionDiT backbone payload ...")
    cosmos_state = dit.state_dict()
    backbone_state = {}
    for k, v in cosmos_state.items():
        # Skip Cosmos-specific module names that don't exist in ActionDiT.
        if k.startswith("patch_embedding."):     # video-only
            continue
        if k.startswith("head."):                # ActionDiT has its own head shape
            continue
        if k.startswith("_freqs_") or k.startswith("freqs"):  # buffers; ActionDiT precomputes its own RoPE
            continue
        backbone_state[k] = v.detach().cpu().to(torch_dtype)

    payload = {
        "backbone_state_dict": backbone_state,
        "meta": {
            "hidden_dim": hidden,
            "ffn_dim": ffn_dim,
            "num_layers": int(cfg.num_layers),
            "num_heads": int(cfg.num_attention_heads),
            "attn_head_dim": int(cfg.attention_head_dim),
            "text_dim": int(cfg.text_embed_dim),
            "freq_dim": 256,
            "eps": 1e-6,
            "source": "cosmos-predict2-2b-video2world",
            "alpha_scaling": "not_applied (dims match exactly with ActionDiT)",
            "notes": "CosmosVideoDiT modulation params kept random — diffusers AdaLN-Lora can't cleanly remap.",
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
