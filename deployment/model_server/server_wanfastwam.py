# Copyright 2025 starVLA community.
"""WebSocket policy server for Wan_FastWAM with FastWAM-official ckpt loaded.

Differs from server_policy.py:
  - Builds the model from a starVLA framework config (yaml), not from a starVLA training ckpt.
  - Strict-loads weights from a FastWAM official release ckpt
    (e.g. libero_uncond_2cam224.pt) using the key mappings defined in
    verify_alignment.py — supports loading the trained MoT (action+video) +
    proprio_encoder all at once.

Usage:
    python deployment/model_server/server_wanfastwam.py \
        --config_yaml examples/LIBERO/train_files/starvla_wanfastwam_libero.yaml \
        --fastwam_ckpt /path/to/libero_uncond_2cam224.pt \
        --wan22_path /path/to/Wan2.2-TI2V-5B-Diffusers \
        --port 10093 \
        --use_bf16
"""

import argparse
import logging
import os
import socket
import sys
import time

import torch
from omegaconf import OmegaConf

# Make this checkout's starVLA win over any installed package; keeps server in sync with our edits.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import build_framework, _auto_import_framework_modules

# verify_alignment lives at repo root
sys.path.insert(0, _REPO)
from verify_alignment import (  # noqa: E402
    remap_fastwam_action_to_ours,
    remap_fastwam_video_to_wanvideo,
)


def main(args) -> None:
    _auto_import_framework_modules()

    cfg = OmegaConf.load(args.config_yaml)
    if args.wan22_path:
        cfg.framework.world_model.base_wm = args.wan22_path
        cfg.framework.qwenvl.base_vlm = args.wan22_path
    logging.info(f"[*] Building Wan_FastWAM (config={args.config_yaml})")
    t0 = time.time()
    model = build_framework(cfg)
    logging.info(f"[*] Built in {time.time()-t0:.1f}s")

    # Two ckpt formats supported:
    #   - FastWAM official: payload = {"mot": <mixtures.{video,action}.* state>, "proprio_encoder": {...}}
    #     → needs key remap (verify_alignment.remap_*)
    #   - starVLA training output: flat state_dict with keys like
    #     `backbone.transformer.*`, `action_expert.*`, `proprio_encoder.*`
    #     → load directly via model.load_state_dict
    logging.info(f"[*] Loading ckpt: {args.fastwam_ckpt}")
    t0 = time.time()
    payload = torch.load(args.fastwam_ckpt, map_location="cpu", weights_only=False)
    logging.info(f"[*] Loaded payload in {time.time()-t0:.1f}s")

    if isinstance(payload, dict) and "mot" in payload:
        # FastWAM-format: needs remap
        logging.info(f"[*] Detected FastWAM format. keys={sorted(payload.keys())}")
        mot = payload["mot"]
        proprio = payload.get("proprio_encoder", {})

        action_state = remap_fastwam_action_to_ours(mot)
        model.action_expert.load_state_dict(action_state, strict=True)
        logging.info(f"  action_expert: strict-loaded ({len(action_state)} tensors)")

        if model.proprio_encoder is not None and proprio:
            model.proprio_encoder.load_state_dict(proprio, strict=True)
            logging.info(f"  proprio_encoder: strict-loaded ({len(proprio)} tensors)")

        video_state, unmapped = remap_fastwam_video_to_wanvideo(mot)
        if unmapped:
            raise RuntimeError(f"Unmappable FastWAM video keys: {unmapped[:5]} ...")
        model.backbone.transformer.load_state_dict(video_state, strict=True)
        logging.info(f"  video diffusers: strict-loaded ({len(video_state)} tensors)")
    else:
        # starVLA training output: flat state_dict, directly compatible.
        # Frozen modules (backbone.text_encoder, backbone.vae, backbone.scheduler)
        # are already initialised from base_wm path, so we only need to load the
        # trainable params.
        logging.info(f"[*] Detected starVLA flat state_dict (n_keys={len(payload)})")
        missing, unexpected = model.load_state_dict(payload, strict=False)
        # Filter out IO-frozen buffers we expect to miss (text_encoder/vae/scheduler init from base_wm).
        important_missing = [
            k for k in missing
            if not (k.startswith("backbone.text_encoder")
                    or k.startswith("backbone.vae")
                    or k.startswith("backbone.scheduler")
                    or k.startswith("backbone.video_processor"))
        ]
        if important_missing:
            raise RuntimeError(
                f"starVLA ckpt missing trainable params (first 5): {important_missing[:5]} "
                f"(total {len(important_missing)} missing)"
            )
        if unexpected:
            logging.warning(f"  ckpt has unexpected keys (first 5): {unexpected[:5]}")
        logging.info(f"  loaded {len(payload) - len(unexpected)} trainable tensors directly")

    # Install state-normalization stats so predict_action normalizes raw eval state
    # to match FastWAM training-time processor.
    if args.dataset_stats:
        import json
        ds = json.load(open(args.dataset_stats))
        # pick first dataset key under 'libero' (our convention). Newer starVLA
        # stats use "state"; older stats used "proprio" for the same vector.
        ds_keys = [k for k in ds.keys() if isinstance(ds[k], dict) and ("proprio" in ds[k] or "state" in ds[k])]
        if ds_keys:
            pp = ds[ds_keys[0]].get("proprio", ds[ds_keys[0]].get("state"))
            model.set_state_stats(pp["min"], pp["max"])
            logging.info(f"[*] state stats installed (key={ds_keys[0]}, dim={len(pp['min'])})")
        else:
            logging.warning(f"[*] dataset_stats has no proprio entry; skipping state norm")

    if args.use_bf16:
        model = model.to(torch.bfloat16)
    model = model.to("cuda").eval()
    logging.info(f"[*] Model on cuda/{model.action_expert.action_encoder.weight.dtype}; ready to serve.")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Creating server on host={hostname} ip={local_ip} port={args.port}")

    server = WebsocketPolicyServer(
        policy=model,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "libero", "framework": "WanFastWAM",
                  "fastwam_ckpt": os.path.basename(args.fastwam_ckpt)},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True,
                        help="starVLA framework config yaml (defines WanFastWAM hyperparams).")
    parser.add_argument("--fastwam_ckpt", type=str, required=True,
                        help="FastWAM official release ckpt (e.g. libero_uncond_2cam224.pt).")
    parser.add_argument("--wan22_path", type=str, default=None,
                        help="Override base_wm/base_vlm path; defaults to whatever is in config_yaml.")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=-1,
                        help="Idle timeout in seconds; -1 means never close.")
    parser.add_argument("--dataset_stats", type=str, default=None,
                        help="Optional path to dataset_statistics.json (starVLA format) "
                             "containing proprio min/max for eval-time state normalization.")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    main(args)
