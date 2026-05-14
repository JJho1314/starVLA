"""Compare HF Diffusers vs FastWAM (DiffSynth-Studio) VAE/T5 outputs.

Quantifies how much the two implementations diverge numerically when fed the
same input. Used to assess whether the libero_10 -4.8% gap could come from
this implementation difference.

Pipeline (sequential on a single GPU to avoid OOM):
    1. Load HF AutoencoderKLWan + UMT5EncoderModel from Wan2.2-TI2V-5B-Diffusers
    2. Encode synthetic / real LIBERO image + prompts → save to CPU
    3. Free GPU
    4. Load FastWAM WanVideoVAE38 + WanTextEncoder via DiffSynth-Studio safetensors
    5. Encode same inputs → save to CPU
    6. Compare element-wise: max-abs-diff, mean-abs-diff, rel-diff, cosine sim

Run:
    cd /data/LFT-W02_data/junjie/VLA_WM/starVLA
    python scripts/compare_vae_t5_hf_vs_fastwam.py \
        --base-wm /data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers \
        --fastwam-ckpts /data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean/checkpoints

For a real LIBERO image, pass ``--real-image /path/to/sample.png``.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_synthetic_video(
    B: int = 1, T: int = 9, H: int = 224, W: int = 448, seed: int = 7
) -> torch.Tensor:
    """Deterministic synthetic video tensor in [-1, 1] (matches video_processor output)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, 3, T, H, W, generator=g) * 2 - 1  # [-1, 1]
    return x.to(torch.bfloat16)


def load_real_video(path: str, T: int = 9, H: int = 224, W: int = 448) -> torch.Tensor:
    """Load a real image; repeat to T frames at (H, W). Returns [1, 3, T, H, W] bf16."""
    img = PILImage.open(path).convert("RGB").resize((W, H), PILImage.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    arr = arr.transpose(2, 0, 1)  # CHW
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(2).repeat(1, 1, T, 1, 1)
    return x.to(torch.bfloat16)


def tensor_diff_stats(a: torch.Tensor, b: torch.Tensor, label: str = "") -> dict:
    a32 = a.detach().to(torch.float32).cpu()
    b32 = b.detach().to(torch.float32).cpu()
    if a32.shape != b32.shape:
        raise ValueError(f"shape mismatch: {tuple(a32.shape)} vs {tuple(b32.shape)}")
    diff = a32 - b32
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    a_norm = a32.norm().item()
    b_norm = b32.norm().item()
    rel_l2 = (diff.norm() / max(a_norm, 1e-8)).item()
    # cosine sim (flatten)
    af, bf = a32.flatten(), b32.flatten()
    cos = torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
    stats = {
        "label": label,
        "shape": tuple(a32.shape),
        "a_norm": a_norm,
        "b_norm": b_norm,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rel_l2_diff": rel_l2,
        "cosine_sim": cos,
    }
    return stats


def print_stats(stats: dict):
    print(f"  [{stats['label']}] shape={stats['shape']}")
    print(f"    a_norm={stats['a_norm']:.4f}  b_norm={stats['b_norm']:.4f}")
    print(f"    max_abs_diff = {stats['max_abs_diff']:.6f}")
    print(f"    mean_abs_diff = {stats['mean_abs_diff']:.6f}")
    print(f"    rel_l2_diff = {stats['rel_l2_diff']*100:.4f}%")
    print(f"    cosine_sim = {stats['cosine_sim']:.8f}")


# ----------------------------------------------------------------------------
# HF path: load + encode
# ----------------------------------------------------------------------------
def run_hf(base_wm: str, video_x: torch.Tensor, prompts: list[str],
           device: str = "cuda:0") -> dict:
    """Load HF VAE + T5, encode, return CPU tensors. Frees GPU on exit."""
    from diffusers import AutoencoderKLWan
    from transformers import T5TokenizerFast, UMT5EncoderModel

    print(f"\n[HF] loading VAE + T5 from {base_wm}")
    vae = AutoencoderKLWan.from_pretrained(
        base_wm, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = T5TokenizerFast.from_pretrained(base_wm, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_wm, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device).eval()

    # VAE encode (apply same diffusers (raw - mean) * (1/std) post-processing as
    # Wan2.WanVideoBackbone._encode_images_vae)
    print("[HF] VAE encoding ...")
    x = video_x.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        raw_latents = vae.encode(x).latent_dist.sample()
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(raw_latents.device, raw_latents.dtype)
    )
    latents_std_inv = (
        1.0 / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(raw_latents.device, raw_latents.dtype)
    )
    hf_latents = ((raw_latents - latents_mean) * latents_std_inv).cpu()
    print(f"[HF] latents: {tuple(hf_latents.shape)} dtype={hf_latents.dtype}")

    # T5 encode
    print("[HF] T5 encoding ...")
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_out = text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        ).last_hidden_state
    hf_text = text_out.to(torch.bfloat16).cpu()
    hf_text_mask = text_inputs.attention_mask.bool().cpu()
    print(f"[HF] text emb: {tuple(hf_text.shape)} mask sum/sample 0: {int(hf_text_mask[0].sum())}")

    # Capture VAE config values used for normalization (so we can apply IDENTICAL post-processing in FastWAM path)
    vae_cfg = {
        "latents_mean": list(vae.config.latents_mean),
        "latents_std": list(vae.config.latents_std),
        "z_dim": int(vae.config.z_dim),
    }

    out = {
        "vae_latents": hf_latents,        # [B, z, T_lat, h, w] bf16, normalized
        "raw_vae_latents": raw_latents.cpu(),  # before mean/std rescale
        "text_emb": hf_text,
        "text_mask": hf_text_mask,
        "vae_config": vae_cfg,
    }

    # Free GPU
    del vae, text_encoder, tokenizer, text_inputs, text_out, raw_latents
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ----------------------------------------------------------------------------
# FastWAM path: load + encode
# ----------------------------------------------------------------------------
def run_fastwam(fastwam_repo: str, fastwam_ckpts: str, video_x: torch.Tensor,
                prompts: list[str], vae_config_from_hf: dict,
                device: str = "cuda:0") -> dict:
    """Load FastWAM VAE + T5 (via DiffSynth loader), encode, return CPU tensors."""
    # Inject FastWAM src
    src = Path(fastwam_repo) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(fastwam_ckpts)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

    from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components

    # Minimal DiT cfg (DiT not loaded — we pass skip_dit_load_from_pretrain=True)
    dit_cfg = {
        "has_image_input": False, "patch_size": (1, 2, 2), "in_dim": 48,
        "hidden_dim": 3072, "ffn_dim": 14336, "freq_dim": 256, "text_dim": 4096,
        "out_dim": 48, "num_heads": 24, "attn_head_dim": 128, "num_layers": 30,
        "eps": 1e-6, "seperated_timestep": True, "require_clip_embedding": False,
        "require_vae_embedding": False, "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False, "action_dim": 7,
        "action_group_causal_mask_mode": "group_diagonal",
    }
    print(f"\n[FastWAM] loading VAE + T5 from {fastwam_ckpts} (DiffSynth .safetensors)")
    components = load_wan22_ti2v_5b_components(
        device=device,
        torch_dtype=torch.bfloat16,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len=128,
        redirect_common_files=True,
        dit_config=dit_cfg,
        skip_dit_load_from_pretrain=True,  # don't load 25GB DiT
        load_text_encoder=True,
    )
    vae = components.vae
    text_encoder = components.text_encoder
    tokenizer = components.tokenizer

    # FastWAM VAE encode: WanVideoVAE38.encode(videos_list, device) — internal
    # `self.scale = [mean, 1/std]` (FastWAM's OWN hard-coded values) is applied
    # inside `model.encode`, so the returned latents are ALREADY normalized.
    # Note: FastWAM's mean/std are NOT identical to HF's vae/config.json values
    # (FastWAM hard-codes its own constants in WanVideoVAE38.__init__). So the
    # returned `fw_normalized` is in FastWAM's normalized space, while HF's
    # `hf_normalized` is in HF's normalized space. We dump both and report:
    #   - fw_normalized vs hf_normalized (apples-to-apples of normalized outputs
    #     each framework uses downstream — this is the diff that matters for
    #     training/inference)
    #   - we also re-normalize fw output to HF's space to isolate layer-math diff
    print("[FastWAM] VAE encoding ...")
    x = video_x.to(device=device, dtype=torch.bfloat16)
    # WanVideoVAE38.encode takes a LIST of [3,T,H,W] (no batch dim per video)
    videos_list = [x[i] for i in range(x.shape[0])]
    with torch.no_grad():
        fw_normalized_t = vae.encode(videos_list, device=device)  # [B, z, T_lat, h, w]
    print(f"[FastWAM] vae.encode -> {tuple(fw_normalized_t.shape)} dtype={fw_normalized_t.dtype}")

    # Recover RAW (pre-norm) latents from FastWAM by inverting:
    #   fw_normalized = (raw - fw_mean) * (1/fw_std)
    # so raw = fw_normalized * fw_std + fw_mean
    fw_mean = vae.mean.to(fw_normalized_t.device, fw_normalized_t.dtype)
    fw_std = vae.std.to(fw_normalized_t.device, fw_normalized_t.dtype)
    raw_latents = (
        fw_normalized_t * fw_std.view(1, -1, 1, 1, 1) + fw_mean.view(1, -1, 1, 1, 1)
    )

    # Re-normalize with HF's mean/std so the comparison isolates LAYER MATH
    # (any remaining diff vs hf normalized output = differences in layer ops).
    hf_mean = (
        torch.tensor(vae_config_from_hf["latents_mean"])
        .view(1, vae_config_from_hf["z_dim"], 1, 1, 1)
        .to(raw_latents.device, raw_latents.dtype)
    )
    hf_std_inv = (
        1.0 / torch.tensor(vae_config_from_hf["latents_std"])
        .view(1, vae_config_from_hf["z_dim"], 1, 1, 1)
        .to(raw_latents.device, raw_latents.dtype)
    )
    fw_renormalized_hf_space = ((raw_latents - hf_mean) * hf_std_inv).cpu()

    fw_latents = fw_normalized_t.cpu()  # FastWAM's native normalized output
    print(f"[FastWAM] normalized latents: {tuple(fw_latents.shape)} dtype={fw_latents.dtype}")

    # FastWAM T5 encode (clean='whitespace' default from loader)
    print("[FastWAM] T5 encoding ...")
    # FastWAM HuggingfaceTokenizer call signature: tok(prompts, return_mask=True, add_special_tokens=...)
    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device).bool()
    with torch.no_grad():
        fw_text = text_encoder(ids, mask)
    fw_text = fw_text.to(torch.bfloat16).cpu()
    fw_text_mask = mask.cpu()
    print(f"[FastWAM] text emb: {tuple(fw_text.shape)} mask sum/sample 0: {int(fw_text_mask[0].sum())}")

    out = {
        "vae_latents": fw_latents,                          # FastWAM native norm space
        "vae_latents_in_hf_space": fw_renormalized_hf_space,  # re-normalized to HF space
        "raw_vae_latents": raw_latents.cpu(),
        "text_emb": fw_text,
        "text_mask": fw_text_mask,
    }

    del vae, text_encoder, tokenizer, components, raw_latents
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-wm",
                    default="/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers",
                    help="HF Diffusers Wan2.2-TI2V-5B folder")
    ap.add_argument("--fastwam-repo",
                    default="/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean",
                    help="FastWAM repo root (must contain src/fastwam/)")
    ap.add_argument("--fastwam-ckpts",
                    default="/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean/checkpoints",
                    help="DIFFSYNTH_MODEL_BASE_PATH (has DiffSynth-Studio/.safetensors)")
    ap.add_argument("--real-image", default=None, help="Optional path to real LIBERO image")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-out", default=None,
                    help="Optional: save dict of {hf, fastwam} encodings to this .pt")
    ap.add_argument("--phase", choices=["hf", "fastwam", "diff", "all"], default="all",
                    help="Run only one phase. Use to keep GPU under tight memory budgets: "
                         "hf → save HF encodings; fastwam → save FastWAM encodings; "
                         "diff → load both .pt files and print diff stats; all → all in one process.")
    ap.add_argument("--hf-out", default="/tmp/hf_encodings.pt",
                    help="(phase=hf) save HF encodings here / (phase=diff) load HF from here")
    ap.add_argument("--fastwam-out", default="/tmp/fastwam_encodings.pt",
                    help="(phase=fastwam) save FastWAM encodings here / (phase=diff) load from here")
    args = ap.parse_args()

    # ---- Inputs ----
    prompts = [
        "A video recorded from a robot's point of view executing the following instruction: pick up the alphabet soup and place it in the basket",
        "A video recorded from a robot's point of view executing the following instruction: open the middle drawer of the cabinet",
        "A video recorded from a robot's point of view executing the following instruction: put the black bowl in the bottom drawer of the cabinet and close it",
    ]

    if args.real_image:
        print(f"Loading real image: {args.real_image}")
        video_x = load_real_video(args.real_image)
    else:
        print("Using synthetic input (rand seed 7)")
        video_x = make_synthetic_video()
    print(f"Input shape: {tuple(video_x.shape)} dtype={video_x.dtype}")

    if args.phase == "hf":
        hf = run_hf(args.base_wm, video_x, prompts, device=args.device)
        torch.save({"video_input": video_x.cpu(), "prompts": prompts, **hf}, args.hf_out)
        print(f"\n[phase=hf] saved encodings to {args.hf_out}")
        return

    if args.phase == "fastwam":
        # Need vae_config from HF for matching normalization; load from saved HF .pt
        if not Path(args.hf_out).exists():
            raise FileNotFoundError(
                f"phase=fastwam needs vae_config from HF phase, but {args.hf_out} missing. "
                "Run --phase=hf first."
            )
        hf_loaded = torch.load(args.hf_out, map_location="cpu", weights_only=False)
        fw = run_fastwam(args.fastwam_repo, args.fastwam_ckpts, video_x, prompts,
                         vae_config_from_hf=hf_loaded["vae_config"], device=args.device)
        torch.save({"video_input": video_x.cpu(), "prompts": prompts, **fw}, args.fastwam_out)
        print(f"\n[phase=fastwam] saved encodings to {args.fastwam_out}")
        return

    if args.phase == "diff":
        hf = torch.load(args.hf_out, map_location="cpu", weights_only=False)
        fw = torch.load(args.fastwam_out, map_location="cpu", weights_only=False)
    else:
        # phase=all (original single-process flow)
        hf = run_hf(args.base_wm, video_x, prompts, device=args.device)
        fw = run_fastwam(args.fastwam_repo, args.fastwam_ckpts, video_x, prompts,
                         vae_config_from_hf=hf["vae_config"], device=args.device)

    # ---- Diff comparison ----
    print("\n" + "=" * 72)
    print("HF vs FastWAM numerical diff (same input, same VAE-norm post-process)")
    print("=" * 72)
    print("\n>>> VAE latents — native normalized output (each in their own normalized space) <<<")
    print("  (FastWAM uses its hard-coded mean/std; HF uses vae/config.json — DIFFERENT values)")
    print_stats(tensor_diff_stats(hf["vae_latents"], fw["vae_latents"], label="vae_latents_native_norm"))

    if "vae_latents_in_hf_space" in fw:
        print("\n>>> VAE latents — FastWAM re-normalized into HF's space (isolates LAYER MATH diff) <<<")
        print_stats(tensor_diff_stats(
            hf["vae_latents"], fw["vae_latents_in_hf_space"],
            label="vae_latents_hf_normspace",
        ))

    print("\n>>> VAE raw latents (pre-normalization) <<<")
    print_stats(tensor_diff_stats(hf["raw_vae_latents"], fw["raw_vae_latents"], label="vae_raw_latents"))

    print("\n>>> Text encoder embeddings <<<")
    print_stats(tensor_diff_stats(hf["text_emb"], fw["text_emb"], label="text_emb_full"))

    # also per-sample (just sample 0 with the real tokens unmasked, both sides)
    for i, prompt in enumerate(prompts):
        n_real_hf = int(hf["text_mask"][i].sum())
        n_real_fw = int(fw["text_mask"][i].sum())
        print(f"\n  prompt {i} (real_tokens HF={n_real_hf}, FW={n_real_fw}): {prompt[:60]!r}...")
        # compare only at real-token positions if they match
        if n_real_hf == n_real_fw:
            print_stats(tensor_diff_stats(
                hf["text_emb"][i, :n_real_hf],
                fw["text_emb"][i, :n_real_fw],
                label=f"text_emb_realtok_p{i}",
            ))
        else:
            print(f"    (skip per-token diff: tokenization mismatch HF={n_real_hf} vs FW={n_real_fw})")

    if args.save_out:
        torch.save({"hf": hf, "fastwam": fw, "video_input": video_x.cpu(), "prompts": prompts},
                   args.save_out)
        print(f"\nSaved encodings to {args.save_out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
