"""Download DiffSynth-Studio's converted .safetensors needed by Wan2_fastwam.

Required files (placed under FASTWAM_CHECKPOINTS_ROOT):
    DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
    DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors

Sources tried, in order:
    1. ModelScope upstream DiffSynth-Studio/Wan-Series-Converted-Safetensors
    2. HuggingFace mirror noodlepop/Wan-Series-Converted-Safetensors

Usage:
    cd /data/LFT-W02_data/junjie/VLA_WM/starVLA
    python scripts/download_fastwam_safetensors.py \\
        --target-root /path/to/FastWAM/checkpoints

On HPC3 (no/limited internet):
    Either run from a node with HF access, OR rsync from this machine where we
    already downloaded everything:
        rsync -avh /data/LFT-W02_data/junjie/weights/DiffSynth-Studio \\
            user@hpc3-login:/data/user/jhe724/workspace/FastWAM/checkpoints/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-root", required=True,
                    help="FASTWAM_CHECKPOINTS_ROOT dir; files go under this/DiffSynth-Studio/...")
    ap.add_argument("--source", choices=["modelscope", "hf", "auto"], default="auto")
    ap.add_argument("--hf-mirror-repo", default="noodlepop/Wan-Series-Converted-Safetensors",
                    help="HF repo id with the same .safetensors files")
    args = ap.parse_args()

    target = Path(args.target_root) / "DiffSynth-Studio" / "Wan-Series-Converted-Safetensors"
    target.mkdir(parents=True, exist_ok=True)
    files = ["Wan2.2_VAE.safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"]

    # Skip files already present
    missing = [f for f in files if not (target / f).exists() or (target / f).stat().st_size < 1_000_000_000]
    if not missing:
        print(f"Both files already present at {target}; nothing to download.")
        return

    def try_modelscope():
        from modelscope import snapshot_download
        snapshot_download(
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            local_dir=str(target),
            allow_file_pattern=missing,
        )

    def try_hf():
        from huggingface_hub import hf_hub_download
        for f in missing:
            print(f"[hf] downloading {f}")
            p = hf_hub_download(repo_id=args.hf_mirror_repo, filename=f, local_dir=str(target))
            print(f"  done: {p}")

    if args.source == "modelscope":
        try_modelscope()
    elif args.source == "hf":
        try_hf()
    else:  # auto
        try:
            print("[auto] trying ModelScope ...")
            try_modelscope()
        except Exception as e:
            print(f"[auto] ModelScope failed: {e}; falling back to HF mirror")
            try_hf()

    # Verify
    for f in files:
        size = (target / f).stat().st_size if (target / f).exists() else 0
        print(f"  {f}: {size/1e9:.2f} GB at {target / f}")
    print(f"All files placed at {target}")


if __name__ == "__main__":
    main()
