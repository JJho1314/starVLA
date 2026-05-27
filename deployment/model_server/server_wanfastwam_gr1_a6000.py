"""WebSocket policy server for Robocasa GR1 WanFastWAM eval on local A6000.

The GR1 ckpt was trained on olabots, so its config bakes in ola paths
(`base_wm: /data/shared/checkpoints/Wan-AI/Wan2.2-TI2V-5B-Diffusers`,
`data_root_dir: /data/shared/...`). On the A6000 box those don't exist, but
the equivalent Wan2.2 weights DO live under /data/LFT-W02_data/junjie/weights.
This wrapper rewrites the config to point at the local copies, installs the
GR1 proprio (state) normalization stats, then serves like the standard
WanFastWAM server.

A6000 has 49 GB, so unlike the lg1 4090 path we do NOT need the text-cache /
CPU-offload tricks — the full 5B video DiT + ActionDiT + live UMT5 + VAE fit
with room to spare, and we keep `load_text_encoder=true` for true zero-cache
inference.

Run:
    python deployment/model_server/server_wanfastwam_gr1_a6000.py \
        --ckpt_path /data/.../gr1_ola_ckpts/pytorch_model.pt \
        --run_dir   /data/.../gr1_ola_ckpts \
        --base_wm   /data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers \
        --diffsynth_base /data/LFT-W02_data/junjie/weights \
        --port 6398 --use_bf16
"""

import argparse
import gc
import json
import logging
import os
import socket
import sys
from pathlib import Path

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config


def _install_state_stats(model, run_dir: str) -> None:
    stats_path = os.path.join(run_dir, "dataset_statistics.json")
    if not os.path.exists(stats_path):
        logging.warning(f"[stats] {stats_path} missing; proprio NOT normalized (SR will drop)")
        return
    with open(stats_path) as f:
        ds = json.load(f)
    chosen = None
    for k, v in ds.items():
        if isinstance(v, dict) and ("state" in v or "proprio" in v):
            chosen = k
            break
    if chosen is None:
        logging.warning(f"[stats] no state/proprio entry in {stats_path}")
        return
    pp = ds[chosen].get("state", ds[chosen].get("proprio"))
    if pp is None or "min" not in pp or "max" not in pp:
        logging.warning(f"[stats] entry under '{chosen}' missing min/max")
        return
    if not hasattr(model, "set_state_stats"):
        logging.warning(f"[stats] model {type(model).__name__} has no set_state_stats()")
        return
    model.set_state_stats(pp["min"], pp["max"])
    logging.info(f"[stats] installed state stats key={chosen} dim={len(pp['min'])}")


def _load_gr1_vla(ckpt_path: str, base_wm: str):
    """from_pretrained with base_wm rewritten to the local Wan2.2 diffusers
    copy. ActionDiT warm-start + transformer diffusers shards are skipped
    (the action ckpt's state_dict re-installs all weights), but base_wm is
    still needed for vae/config.json norm stats."""
    ckpt_p = Path(ckpt_path)
    model_config, norm_stats = read_mode_config(ckpt_p)

    wm = model_config.get("framework", {}).get("world_model", {})
    if isinstance(wm, dict):
        old = wm.get("base_wm")
        wm["base_wm"] = base_wm
        wm["skip_transformer_load"] = True   # ckpt state_dict has the DiT weights
        wm["load_text_encoder"] = True       # A6000 has room for live UMT5
        wm["text_embed_cache_path"] = None
        logging.info(f"[gr1] base_wm {old} -> {base_wm}")

    adit = model_config.get("framework", {}).get("action_dit", {})
    if isinstance(adit, dict):
        adit["pretrained_path"] = None
        adit["skip_pretrained_load"] = True

    # Also retarget the legacy qwenvl.base_vlm if it points at base_wm.
    qv = model_config.get("framework", {}).get("qwenvl", {})
    if isinstance(qv, dict) and qv.get("base_vlm", "").endswith("Wan2.2-TI2V-5B-Diffusers"):
        qv["base_vlm"] = base_wm

    cfg = dict_to_namespace(model_config)
    cfg.trainer.pretrained_checkpoint = None
    model = build_framework(cfg=cfg)
    model.norm_stats = norm_stats

    if ckpt_p.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(ckpt_p))
    else:
        state = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    return model


def main(args) -> None:
    # Point DiffSynth (VAE + UMT5 .safetensors) at the local weights tree.
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", args.diffsynth_base)
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

    logging.info(f"Loading GR1 WanFastWAM from {args.ckpt_path}")
    vla = _load_gr1_vla(args.ckpt_path, args.base_wm)
    # GR1 does NOT use server-side min/max proprio normalization: the robocasa
    # client (model2robocasa_interface.normalize_state) sin/cos-encodes the raw
    # 29-dim joint state into the 58-dim vector the proprio_encoder expects
    # (proprio_encoder.weight is [4096, 58]). Installing the 29-dim dataset
    # min/max stats here makes the model's _normalize_state try to broadcast
    # 29-dim stats over the 58-dim input -> "size of tensor a (58) must match
    # b (29)" crash. So we intentionally skip set_state_stats for GR1.
    if args.install_state_stats:
        _install_state_stats(vla, args.run_dir)
    else:
        logging.info("[gr1] skipping set_state_stats (client sin/cos-encodes "
                     "state to 58-dim; server must pass through, not re-normalize)")

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
    torch.cuda.empty_cache()

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"
    logging.info(f"Creating server (host={hostname} ip={local_ip} port={args.port})")

    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "robocasa_gr1", "ckpt": args.ckpt_path},
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--run_dir", required=True,
                   help="dir with dataset_statistics.json (for GR1 proprio stats)")
    p.add_argument("--base_wm", default="/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--diffsynth_base", default="/data/LFT-W02_data/junjie/weights")
    p.add_argument("--port", type=int, default=6398)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--idle_timeout", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_wait_ms", type=int, default=20)
    p.add_argument("--install_state_stats", action="store_true",
                   help="GR1 default OFF: client sin/cos-encodes state to 58-dim; "
                        "server-side 29-dim min/max would crash. Only set if a ckpt "
                        "truly expects raw+minmax proprio.")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(build_argparser().parse_args())
