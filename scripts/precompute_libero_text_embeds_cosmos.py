"""Pre-compute Cosmos T5 (T5XXL, 1024-dim) embeddings for LIBERO task instructions.

Sibling of ``precompute_libero_text_embeds.py`` which targets Wan UMT5-XXL
(4096-dim). Cosmos-Predict2-2B-Video2World ships a T5EncoderModel with
hidden_dim=1024, max_length=512 (Cosmos default — different from Wan's 128).
Embeds produced here are NOT interchangeable with the Wan cache.

Output: ``.pt`` file mapping ``{task_string: {"embed": [512, 1024] bf16,
"mask": [512] bool}}``. Loaded by ``CosmoPredict_fastwam`` when
``framework.world_model.text_embed_cache_path`` is set.

Use:
    python scripts/precompute_libero_text_embeds_cosmos.py \\
        --cosmos_path /data/.../Cosmos-Predict2-2B-Video2World \\
        --libero_root /data/.../LIBERO-fastwam \\
        --suites goal,object,spatial,10 \\
        --template "A video recorded from a robot's point of view executing the following instruction: {task}" \\
        --out /data/.../libero_cosmos_t5_embeds.pt

The ``--template`` MUST match ``framework.world_model.text_prompt_template`` in
the training yaml — cache lookup is by the wrapped string.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5TokenizerFast


def collect_task_strings(libero_root: Path, suites: list[str]) -> list[str]:
    """Walk ``libero_<suite>_no_noops_lerobot/meta/tasks.jsonl`` for each suite.
    Return the union of unique ``task`` strings (sorted for determinism)."""
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
    ap.add_argument("--cosmos_path", required=True,
                    help="Path to Cosmos-Predict2-2B-Video2World (containing tokenizer/ + text_encoder/).")
    ap.add_argument("--libero_root", required=True,
                    help="Root containing `libero_<suite>_no_noops_lerobot/` directories.")
    ap.add_argument("--suites", default="goal,object,spatial,10")
    ap.add_argument("--max_length", type=int, default=512,
                    help="Cosmos T5 max length (Cosmos default 512, NOT Wan UMT5's 128).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=None,
                    help="Prompt wrapper with `{task}` (matches text_prompt_template in yaml).")
    args = ap.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    libero_root = Path(args.libero_root)

    tasks = collect_task_strings(libero_root, suites)
    print(f"[1/3] Collected {len(tasks)} unique task strings across suites={suites}")
    for t in tasks:
        print(f"    - {t!r}")

    print(f"[2/3] Loading Cosmos T5 from {args.cosmos_path} ...")
    device = torch.device(args.device)
    tokenizer = T5TokenizerFast.from_pretrained(args.cosmos_path, subfolder="tokenizer")
    encoder = T5EncoderModel.from_pretrained(
        args.cosmos_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device).eval()

    cache: dict[str, dict[str, torch.Tensor]] = {}
    print(f"[3/3] Encoding {len(tasks)} tasks at max_length={args.max_length} "
          f"(template={'<wrapper>' if args.template else 'raw'}) ...")
    with torch.no_grad():
        for task in tasks:
            text = args.template.format(task=task) if args.template else task
            inputs = tokenizer(
                [text],
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
            ).last_hidden_state    # [1, L, 1024]
            cache[text] = {
                "embed": embeds[0].to(dtype=torch.bfloat16).cpu(),  # [L, 1024]
                "mask": inputs.attention_mask[0].to(dtype=torch.bool).cpu(),
            }
            n_real = int(inputs.attention_mask.sum().item())
            print(f"    {text!r}: {n_real} real tokens (hidden_dim={embeds.shape[-1]})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "max_length": args.max_length,
            "cosmos_path": args.cosmos_path,
            "tasks": tasks,
            "cache": cache,
        },
        out_path,
    )
    print(f"Saved {len(cache)} entries to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
