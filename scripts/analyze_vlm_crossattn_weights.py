"""Quantify how much the DiT cross-attention actually relies on the VLM-query
segment vs the T5-text segment vs the proprio segment.

Context layout reaching every ``cross_attn`` (both video DiT and ActionDiT) is::

    context = text_embedding([ T5_text (L_t) | VLM_query (N) | proprio (P) ])

(`WanMetaQueryFastWAM` concatenates the N MetaQuery tokens after the UMT5 text
embeds; `_inject_proprio` then appends P proprio tokens. `text_embedding` is a
per-token Linear, so ordering is preserved.)

`cross_attn` internally calls `F.scaled_dot_product_attention`, which does NOT
expose attention weights. So we register a forward-pre-hook on each `cross_attn`
module, recompute ``softmax(q·kᵀ / sqrt(d) + mask)`` with the module's own
q/k/norm_q/norm_k weights, and sum the probability mass landing on each segment.

Interpretation:
    * mass(VLM) is the fraction of cross-attention that video/action tokens send
      to the 64 MetaQuery tokens, averaged over heads and query positions.
    * If mass(VLM) < ~5% across most layers, the VLM channel is barely used →
      connector too shallow or freeze-plan didn't unlock enough. Deepen the
      connector (8 → 16/24) or unfreeze more, then re-measure.

Run (mirrors scripts/smoke_wan_metaquery_fastwam.py for model building)::

    CUDA_VISIBLE_DEVICES=0 python scripts/analyze_vlm_crossattn_weights.py \
        --config examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero_local.yaml \
        --state-dict /path/to/trained_model.safetensors \
        --image /path/to/first_frame.png \
        --instruction "pick up the red block and place it in the basket" \
        --out-json vlm_crossattn.json --out-plot vlm_crossattn.png

Notes:
    * Without --state-dict the connector is RANDOM-INIT; the numbers then reflect
      an untrained model. Always point at a trained checkpoint to judge "does the
      VLM channel matter".
    * Memory: full [B, heads, Sv, L] score tensors are large for video. We
      subsample query tokens to --max-query-tokens (default 1024) per layer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.base_framework import build_framework


# --------------------------------------------------------------------------- #
# Batch construction (schema must match WanFastWAM.forward())
# --------------------------------------------------------------------------- #
def _load_frame(image_path: str | None, hw: int) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB").resize((hw, hw))
    arr = np.random.randint(0, 255, (hw, hw, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_batch(num_frames: int, chunk_len: int, action_dim: int, state_dim: int,
               image_path: str | None, instruction: str, hw: int = 224):
    """One VLA example. The conditioning frame (frame[0]) is what the VLM sees;
    duplicate it across the T-frame video stub so the backbone VAE is happy."""
    frame = _load_frame(image_path, hw)
    cam_pri = [frame for _ in range(num_frames)]
    cam_wri = [frame for _ in range(num_frames)]
    example = {
        "image": [cam_pri, cam_wri],                       # 2 cams, T frames each
        "lang": instruction,
        "action": np.random.randn(chunk_len, action_dim).astype(np.float32),
        "state": np.zeros((state_dim,), dtype=np.float32),
        "action_is_pad": np.zeros((chunk_len,), dtype=bool),
        "image_is_pad": np.zeros((num_frames,), dtype=bool),
    }
    return [example]


# --------------------------------------------------------------------------- #
# Cross-attention probe
# --------------------------------------------------------------------------- #
class CrossAttnProbe:
    """Registers forward-pre-hooks on cross_attn modules and accumulates the
    per-segment attention mass each time they fire."""

    def __init__(self, num_queries: int, num_proprio: int, max_query_tokens: int,
                 seed: int = 0):
        self.num_queries = int(num_queries)
        self.num_proprio = int(num_proprio)
        self.max_query_tokens = int(max_query_tokens)
        self.gen = torch.Generator().manual_seed(seed)
        # records[(expert, layer)] = dict(t5=.., vlm=.., proprio=.., L=.., Lt=.., Sq=..)
        self.records: dict = {}
        self.handles: list = []
        self._printed_layout = False

    def _hook(self, expert: str, layer: int):
        def hook(module, args, kwargs):
            # cross_attn(x, ctx, ctx_mask=...) — see CrossAttention.forward
            x = args[0]
            ctx = args[1] if len(args) > 1 else kwargs["ctx"]
            ctx_mask = kwargs.get("ctx_mask", args[2] if len(args) > 2 else None)

            B, Sq, _ = x.shape
            L = ctx.shape[1]
            n = int(module.num_heads)
            d = int(module.attn_head_dim)

            L_t = L - self.num_queries - self.num_proprio
            if L_t <= 0:
                raise RuntimeError(
                    f"[{expert} L{layer}] resolved T5 length L_t={L_t} <= 0 "
                    f"(L={L}, N={self.num_queries}, P={self.num_proprio}). "
                    f"Check --num-proprio / --num-queries."
                )
            if not self._printed_layout:
                print(f"[layout] context L={L}  ->  T5[0:{L_t}] | "
                      f"VLM[{L_t}:{L_t + self.num_queries}] | "
                      f"proprio[{L_t + self.num_queries}:{L}]   (expert={expert})")
                self._printed_layout = True

            # Subsample query positions for memory.
            if Sq > self.max_query_tokens:
                idx = torch.randperm(Sq, generator=self.gen)[: self.max_query_tokens]
                idx = idx.to(x.device)
                xq = x.index_select(1, idx)
            else:
                idx = None
                xq = x

            with torch.no_grad():
                q = module.norm_q(module.q(xq)).float()      # [B, Sq', n*d]
                k = module.norm_k(module.k(ctx)).float()     # [B, L,  n*d]
                Sqs = q.shape[1]
                qh = q.view(B, Sqs, n, d).permute(0, 2, 1, 3)   # [B, n, Sq', d]
                kh = k.view(B, L, n, d).permute(0, 2, 1, 3)     # [B, n, L,  d]
                scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(d)  # [B,n,Sq',L]

                if ctx_mask is not None:
                    m = ctx_mask
                    if m.dim() == 3:
                        m = m.unsqueeze(1)                    # [B,1,Sq,L]
                    if idx is not None and m.shape[2] == Sq:
                        m = m.index_select(2, idx)
                    if m.dtype == torch.bool:
                        scores = scores.masked_fill(~m, float("-inf"))
                    else:
                        scores = scores + m.float()

                probs = scores.softmax(dim=-1)               # [B, n, Sq', L]
                # fraction of mass to each segment, averaged over (batch, heads, queries)
                t5 = probs[..., :L_t].sum(-1).mean().item()
                vlm = probs[..., L_t:L_t + self.num_queries].sum(-1).mean().item()
                pro = probs[..., L_t + self.num_queries:].sum(-1).mean().item()

            self.records[(expert, layer)] = {
                "t5": t5, "vlm": vlm, "proprio": pro,
                "L": L, "L_t": L_t, "Sq": Sq,
            }
        return hook

    def attach(self, model):
        # video expert == backbone.transformer ; action expert == action_expert
        video_blocks = model.backbone.transformer.blocks
        for i, blk in enumerate(video_blocks):
            self.handles.append(
                blk.cross_attn.register_forward_pre_hook(
                    self._hook("video", i), with_kwargs=True
                )
            )
        action_blocks = getattr(model, "action_expert", None)
        if action_blocks is not None:
            for i, blk in enumerate(model.action_expert.blocks):
                self.handles.append(
                    blk.cross_attn.register_forward_pre_hook(
                        self._hook("action", i), with_kwargs=True
                    )
                )
        print(f"[probe] attached to {len(video_blocks)} video + "
              f"{0 if action_blocks is None else len(model.action_expert.blocks)} "
              f"action cross_attn layers")

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # ----------------------------- reporting ----------------------------- #
    def report(self):
        by_expert = defaultdict(list)
        for (expert, layer), rec in sorted(self.records.items()):
            by_expert[expert].append((layer, rec))

        summary = {}
        for expert, rows in by_expert.items():
            print(f"\n=== expert: {expert} "
                  f"(per-layer cross-attn mass; T5 / VLM / proprio) ===")
            print(f"{'layer':>5} | {'T5':>7} | {'VLM':>7} | {'proprio':>7}")
            print("-" * 36)
            vlm_vals = []
            for layer, rec in rows:
                vlm_vals.append(rec["vlm"])
                print(f"{layer:>5} | {rec['t5']*100:6.2f}% | "
                      f"{rec['vlm']*100:6.2f}% | {rec['proprio']*100:6.2f}%")
            vlm_arr = np.array(vlm_vals)
            mean_vlm = float(vlm_arr.mean())
            print("-" * 36)
            print(f"  VLM mass: mean={mean_vlm*100:.2f}%  "
                  f"min={vlm_arr.min()*100:.2f}%  max={vlm_arr.max()*100:.2f}%  "
                  f"(layers<5%: {int((vlm_arr < 0.05).sum())}/{len(vlm_arr)})")
            verdict = ("⚠️  VLM channel WEAK (<5% mean) — deepen connector or "
                       "unlock freeze plan" if mean_vlm < 0.05
                       else "✅ VLM channel is being used")
            print(f"  verdict: {verdict}")
            summary[expert] = {
                "mean_vlm": mean_vlm,
                "min_vlm": float(vlm_arr.min()),
                "max_vlm": float(vlm_arr.max()),
                "layers_below_5pct": int((vlm_arr < 0.05).sum()),
                "num_layers": len(vlm_arr),
            }
        return summary

    def to_json(self, path: str, summary: dict, meta: dict):
        out = {
            "meta": meta,
            "summary": summary,
            "per_layer": {
                f"{e}.{l}": r for (e, l), r in sorted(self.records.items())
            },
        }
        Path(path).write_text(json.dumps(out, indent=2))
        print(f"\n[probe] wrote {path}")

    def plot(self, path: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001
            print(f"[probe] matplotlib unavailable ({e}); skipping plot")
            return

        by_expert = defaultdict(list)
        for (expert, layer), rec in sorted(self.records.items()):
            by_expert[expert].append((layer, rec))

        fig, axes = plt.subplots(len(by_expert), 1,
                                 figsize=(11, 3.2 * len(by_expert)), squeeze=False)
        for ax, (expert, rows) in zip(axes[:, 0], by_expert.items()):
            layers = [l for l, _ in rows]
            t5 = np.array([r["t5"] for _, r in rows])
            vlm = np.array([r["vlm"] for _, r in rows])
            pro = np.array([r["proprio"] for _, r in rows])
            ax.bar(layers, t5, label="T5 text", color="#4C72B0")
            ax.bar(layers, vlm, bottom=t5, label="VLM query", color="#DD8452")
            ax.bar(layers, pro, bottom=t5 + vlm, label="proprio", color="#55A868")
            ax.axhline(0.05, ls="--", lw=0.8, color="red", alpha=0.6)
            ax.set_title(f"{expert} expert — cross-attn mass per layer")
            ax.set_xlabel("DiT layer")
            ax.set_ylabel("attention mass")
            ax.set_ylim(0, 1)
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        print(f"[probe] wrote {path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero_local.yaml",
    )
    ap.add_argument("--state-dict", default=None,
                    help="Optional trained model weights (.safetensors or .pt). "
                         "Loaded with strict=False over build_framework(config).")
    ap.add_argument("--image", default=None, help="Real first-frame image; random if omitted.")
    ap.add_argument("--instruction", default="pick up the red block and place it in the basket")
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--num-proprio", type=int, default=1,
                    help="proprio tokens appended after VLM queries (1 if state injected, else 0).")
    ap.add_argument("--max-query-tokens", type=int, default=1024)
    ap.add_argument("--out-json", default="vlm_crossattn.json")
    ap.add_argument("--out-plot", default="vlm_crossattn.png")
    args = ap.parse_args()

    print(f"[main] loading config: {args.config}")
    cfg = OmegaConf.load(args.config)
    print(f"[main] building framework {cfg.framework.name} …")
    model = build_framework(cfg)

    if args.state_dict:
        print(f"[main] loading state_dict: {args.state_dict}")
        if args.state_dict.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(args.state_dict)
        else:
            sd = torch.load(args.state_dict, map_location="cpu")
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Only complain about connector / vlm-token rows missing — those are the
        # parts whose training we are trying to evaluate.
        key_missing = [k for k in missing if "connector" in k or "vlm" in k]
        if key_missing:
            print(f"[main] WARNING: {len(key_missing)} connector/vlm keys MISSING "
                  f"from state_dict (showing 5): {key_missing[:5]}")
        print(f"[main] loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("[main] WARNING: no --state-dict → connector is RANDOM-INIT; "
              "numbers reflect an untrained model.")

    model = model.to(dtype=torch.bfloat16, device="cuda").eval()
    num_queries = int(getattr(model, "num_queries", 64))
    print(f"[main] num_queries={num_queries}, num_proprio={args.num_proprio}")

    batch = make_batch(
        num_frames=args.num_frames,
        chunk_len=model.chunk_len,
        action_dim=model.action_dim,
        state_dim=int(getattr(model, "state_dim", 8)),
        image_path=args.image,
        instruction=args.instruction,
    )

    probe = CrossAttnProbe(
        num_queries=num_queries,
        num_proprio=args.num_proprio,
        max_query_tokens=args.max_query_tokens,
    )
    probe.attach(model)

    print("[main] running forward …")
    with torch.no_grad():
        _ = model(batch)

    probe.detach()
    summary = probe.report()
    meta = {
        "config": args.config,
        "state_dict": args.state_dict,
        "image": args.image,
        "instruction": args.instruction,
        "num_queries": num_queries,
        "num_proprio": args.num_proprio,
        "num_frames": args.num_frames,
    }
    probe.to_json(args.out_json, summary, meta)
    probe.plot(args.out_plot)


if __name__ == "__main__":
    main()
