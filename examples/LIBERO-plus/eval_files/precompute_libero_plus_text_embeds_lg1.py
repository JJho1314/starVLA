"""Pre-compute UMT5-XXL embeddings for all LIBERO-plus task descriptions.

Why: on a 24 GB 4090 the full Wan2.2 + ActionDiT + UMT5-XXL + VAE stack
barely fits — UMT5 alone is ~11 GB in bf16. Dropping UMT5 frees enough
headroom for the MoT forward; we replace its live encoder with a per-task
cache lookup.

Cache format matches `Wan2_fastwam._load_text_context_from_dir`:
    key      = template.format(task=raw_task)   # whitespace stripped
    hashed   = sha256(key.utf8).hexdigest()
    filename = f"{hashed}.t5_len{max_length}.wan22ti2v5b.pt"
    payload  = {"context": Tensor[L, 4096] bf16, "mask": Tensor[L] bool}

Run on lg1 (single GPU, ~17 min for ~10k tasks):
    CUDA_VISIBLE_DEVICES=0 python examples/LIBERO-plus/eval_files/precompute_libero_plus_text_embeds_lg1.py \
        --out /data3/junjie/eval_runs/libero_plus_text_cache_lg1 \
        --suites libero_spatial,libero_object,libero_goal,libero_10
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _inject_fastwam_src():
    repo = os.environ.get("STARVLA_FASTWAM_REPO_PATH")
    if not repo:
        raise RuntimeError("STARVLA_FASTWAM_REPO_PATH not set")
    src = Path(repo) / "src"
    if not src.is_dir():
        raise RuntimeError(f"{src} not a directory")
    sys.path.insert(0, str(src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir for hashed .pt files.")
    ap.add_argument(
        "--suites",
        default="libero_spatial,libero_object,libero_goal,libero_10",
        help="Comma-separated LIBERO-plus suites to walk.",
    )
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--template",
        default="A video recorded from a robot's point of view executing the following instruction: {task}",
        help="Prompt template; '{task}' is replaced with the bddl task description. "
             "Must match `framework.world_model.text_prompt_template` from training.",
    )
    args = ap.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Walk LIBERO-plus benchmark for all task descriptions.
    print(f"[1/4] Walking LIBERO-plus suites {suites} ...")
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    all_tasks: set[str] = set()
    for s in suites:
        suite = bd[s]()
        for tid in range(suite.n_tasks):
            t = suite.get_task(tid)
            if t.language:
                all_tasks.add(t.language.strip())
    print(f"    Collected {len(all_tasks)} unique task strings.")

    # Load T5 + tokenizer using the FastWAM loader (uses .safetensors paths).
    print("[2/4] Loading UMT5-XXL via FastWAM loader ...")
    _inject_fastwam_src()
    from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components

    # Loader requires a dit_config even when DiT is skipped — pass canonical
    # Wan2.2-TI2V-5B defaults from Wan2_fastwam._build_video_dit_config.
    dit_dummy_cfg = {
        "has_image_input": False, "patch_size": (1, 2, 2), "in_dim": 48,
        "hidden_dim": 3072, "ffn_dim": 14336, "freq_dim": 256, "text_dim": 4096,
        "out_dim": 48, "num_heads": 24, "attn_head_dim": 128, "num_layers": 30,
        "eps": 1e-6, "seperated_timestep": True, "require_clip_embedding": False,
        "require_vae_embedding": False, "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": False, "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False, "action_dim": 7,
        "action_group_causal_mask_mode": "group_diagonal",
    }
    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch.bfloat16,
        dit_config=dit_dummy_cfg,
        skip_dit_load_from_pretrain=True,   # only need text encoder
        load_text_encoder=True,
    )
    text_encoder = components.text_encoder
    tokenizer = components.tokenizer
    assert text_encoder is not None and tokenizer is not None, "T5 loader failed"
    text_encoder = text_encoder.eval()
    device = torch.device(args.device)

    # Encode each task, hash, save.
    print(f"[3/4] Encoding {len(all_tasks)} tasks and writing cache to {out_dir} ...")
    skipped = saved = 0
    for raw_task in tqdm.tqdm(sorted(all_tasks)):
        key = args.template.format(task=raw_task)
        hashed = hashlib.sha256(key.encode("utf-8")).hexdigest()
        fname = f"{hashed}.t5_len{args.max_length}.wan22ti2v5b.pt"
        out_path = out_dir / fname
        if out_path.exists():
            skipped += 1
            continue

        # HuggingfaceTokenizer.__call__ returns (ids, mask)
        ids, mask = tokenizer([key], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            context = text_encoder(ids, mask)   # [1, L, 4096]
        torch.save(
            {
                "context": context[0].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
                "mask": mask[0].detach().to("cpu", dtype=torch.bool).contiguous(),
            },
            str(out_path),
        )
        saved += 1
    print(f"[4/4] Done. saved={saved} skipped={skipped} total={len(all_tasks)}")
    print(f"    Cache dir: {out_dir}")
    print(f"    Set framework.world_model.text_embed_cache_path = {out_dir}")


if __name__ == "__main__":
    main()
