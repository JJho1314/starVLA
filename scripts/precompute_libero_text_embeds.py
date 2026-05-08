"""Pre-compute UMT5-XXL embeddings for all unique LIBERO task instructions.

Why: starVLA's WanFastWAM loads UMT5-XXL (~11 GB in bf16) on every GPU at training
time, which on H100-80GB blocks ZeRO-1 + bs=16 (FastWAM's per-device batch). LIBERO
has ~40 unique task strings across 4 suites — caching their UMT5 outputs to disk
costs ~5 MB per string and skips loading the encoder entirely at training time.

Output: a single `.pt` file mapping `{task_string: {"embed": [128, 4096] bf16,
"mask": [128] bool}}`. Loaded by `Wan2.py` when `text_embed_cache_path` is set.

Run:
    python scripts/precompute_libero_text_embeds.py \
        --wan22_path /data/.../Wan2.2-TI2V-5B-Diffusers \
        --libero_root /data/.../LIBERO-fastwam \
        --suites goal,object,spatial,10 \
        --out /data/.../LIBERO-fastwam/libero_umt5_text_embeds.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import T5TokenizerFast, UMT5EncoderModel


def collect_task_strings(libero_root: Path, suites: list[str]) -> list[str]:
    """Walk `libero_root/libero_<suite>_no_noops_lerobot/meta/tasks.jsonl` for
    each suite, return the union of unique `task` strings (sorted for determinism)."""
    seen: set[str] = set()
    for suite in suites:
        meta = libero_root / f"libero_{suite}_no_noops_lerobot" / "meta" / "tasks.jsonl"
        if not meta.exists():
            print(f"  WARN: missing {meta}")
            continue
        with open(meta) as f:
            for line in f:
                rec = json.loads(line)
                task = rec.get("task")
                if task:
                    seen.add(task.strip())
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan22_path", required=True,
                    help="Path to Wan2.2-TI2V-5B-Diffusers (containing tokenizer/ + text_encoder/).")
    ap.add_argument("--libero_root", required=True,
                    help="Root containing `libero_<suite>_no_noops_lerobot/` directories.")
    ap.add_argument("--suites", default="goal,object,spatial,10",
                    help="Comma-separated suite names to walk.")
    ap.add_argument("--max_length", type=int, default=128,
                    help="UMT5 token sequence length (must match training context_len).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="Output .pt file path.")
    args = ap.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    libero_root = Path(args.libero_root)

    tasks = collect_task_strings(libero_root, suites)
    print(f"[1/3] Collected {len(tasks)} unique task strings across suites={suites}")
    for t in tasks:
        print(f"    - {t!r}")

    print(f"[2/3] Loading UMT5-XXL from {args.wan22_path} ...")
    device = torch.device(args.device)
    tokenizer = T5TokenizerFast.from_pretrained(args.wan22_path, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        args.wan22_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device).eval()

    # Encode each task, save as dict.
    cache: dict[str, dict[str, torch.Tensor]] = {}
    print(f"[3/3] Encoding {len(tasks)} tasks at max_length={args.max_length} ...")
    with torch.no_grad():
        for task in tasks:
            inputs = tokenizer(
                [task],
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device)
            embeds = encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            ).last_hidden_state
            cache[task] = {
                "embed": embeds[0].to(dtype=torch.bfloat16).cpu(),  # [L, D]
                "mask": inputs.attention_mask[0].to(dtype=torch.bool).cpu(),  # [L]
            }
            n_real = int(inputs.attention_mask.sum().item())
            print(f"    {task!r}: {n_real} real tokens")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "max_length": args.max_length,
            "wan22_path": args.wan22_path,
            "tasks": tasks,
            "cache": cache,
        },
        out_path,
    )
    print(f"Saved {len(cache)} entries to {out_path} (size {out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
