"""Convert starVLA-trained Wan_FastWAM checkpoint to FastWAM official ckpt format.

Post-C-refactor starVLA model state_dict layout (FastWAM-flavour throughout):
  - backbone.text_encoder.* (frozen, drop)
  - backbone.vae.*          (frozen, drop)
  - backbone.transformer.*  → mixtures.video.*    (1:1 prefix substitution; same keys as FastWAM's WanVideoDiT)
  - action_expert.*         → mixtures.action.*   (1:1 prefix substitution; same keys as FastWAM's ActionDiT)
  - proprio_encoder.*       → proprio_encoder.{weight,bias} (top-level)

After the C refactor (vendored FastWAM `wan_video_dit.py` + `mot.py`), the
diffusers→FastWAM keymap rename is no longer needed — starVLA's
`backbone.transformer` IS the FastWAM `WanVideoDiT`, with identical key names.
This file is now just a prefix swap + slim-down.
"""
import argparse
import os

import torch


def convert(src_ckpt: str, dst_ckpt: str, step: int):
    src = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(src, dict):
        raise ValueError(f"expected dict-like ckpt, got {type(src)}")

    mot = {}
    proprio = {}
    dropped = {"text_encoder": 0, "vae": 0, "other": 0}
    unmapped = []

    for k, v in src.items():
        if k.startswith("action_expert."):
            mot["mixtures.action." + k[len("action_expert."):]] = v
        elif k.startswith("backbone.transformer."):
            mot["mixtures.video." + k[len("backbone.transformer."):]] = v
        elif k.startswith("proprio_encoder."):
            proprio[k[len("proprio_encoder."):]] = v
        elif k.startswith("backbone.text_encoder."):
            dropped["text_encoder"] += 1
        elif k.startswith("backbone.vae."):
            dropped["vae"] += 1
        else:
            dropped["other"] += 1
            unmapped.append(k)

    print(f"[*] mot keys mapped:   {len(mot)} (expected 1649: 824 action + 825 video)")
    print(f"[*] proprio keys:      {sorted(proprio.keys())}")
    print(f"[*] dropped:           {dropped}")
    if unmapped:
        print(f"[!] UNMAPPED ({len(unmapped)}):")
        for k in unmapped[:20]:
            print(f"    {k}")
        raise RuntimeError(f"{len(unmapped)} unmapped keys; aborting")

    if len(mot) != 1649:
        raise RuntimeError(f"expected 1649 mot keys, got {len(mot)}")
    if sorted(proprio.keys()) != ["bias", "weight"]:
        raise RuntimeError(f"unexpected proprio keys: {sorted(proprio.keys())}")

    out = {
        "mot": mot,
        "step": int(step),
        "torch_dtype": "bfloat16",
        "proprio_encoder": proprio,
    }
    os.makedirs(os.path.dirname(dst_ckpt), exist_ok=True)
    torch.save(out, dst_ckpt)
    sz = os.path.getsize(dst_ckpt) / 1024**3
    print(f"[+] saved {dst_ckpt} ({sz:.2f} GiB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="starVLA training ckpt (e.g. steps_20000_pytorch_model.pt)")
    p.add_argument("--dst", required=True, help="output FastWAM-format ckpt")
    p.add_argument("--step", type=int, default=0, help="step value to embed in ckpt (informational)")
    args = p.parse_args()
    convert(args.src, args.dst, args.step)


if __name__ == "__main__":
    main()
