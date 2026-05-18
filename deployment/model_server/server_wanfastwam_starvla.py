"""WebSocket server for WanFastWAM trained via starVLA (starVLA-native ckpt).

Differs from server_policy.py: ALSO installs state min/max from dataset_statistics.json
so the framework's `_normalize_state` works at inference. Differs from
server_wanfastwam.py: does NOT key-map a FastWAM-official ckpt; instead expects
a starVLA-native `pytorch_model.pt` saved by our trainer.

Required dir layout (passed via --run_dir):
    <run_dir>/
        final_model/pytorch_model.pt   (or another path via --ckpt_path)
        config.yaml
        dataset_statistics.json

Usage:
    python deployment/model_server/server_wanfastwam_starvla.py \
        --run_dir /path/to/fwalign_olabots_ckpts \
        --port 6694 --use_bf16 --idle_timeout -1
"""

import argparse
import json
import logging
import os
import socket
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


def _install_state_stats(model, run_dir: str) -> None:
    stats_path = os.path.join(run_dir, "dataset_statistics.json")
    if not os.path.exists(stats_path):
        logging.warning(f"[stats] {stats_path} not found; skipping set_state_stats")
        return
    with open(stats_path) as f:
        ds = json.load(f)
    # pick the first key that has a state/proprio block (e.g. "franka", "libero", etc.)
    chosen = None
    for k, v in ds.items():
        if isinstance(v, dict) and ("state" in v or "proprio" in v):
            chosen = k
            break
    if chosen is None:
        logging.warning(f"[stats] no state/proprio entry in {stats_path}; skipping")
        return
    pp = ds[chosen].get("state", ds[chosen].get("proprio"))
    if pp is None or "min" not in pp or "max" not in pp:
        logging.warning(f"[stats] entry under '{chosen}' missing min/max; skipping")
        return
    if not hasattr(model, "set_state_stats"):
        logging.warning(f"[stats] model {type(model).__name__} has no set_state_stats(); skipping")
        return
    model.set_state_stats(pp["min"], pp["max"])
    logging.info(f"[stats] installed state stats (key={chosen}, dim={len(pp['min'])})")


def main(args) -> None:
    ckpt_path = args.ckpt_path or os.path.join(args.run_dir, "final_model", "pytorch_model.pt")
    logging.info(f"Loading framework from {ckpt_path}")
    vla = baseframework.from_pretrained(ckpt_path)

    _install_state_stats(vla, args.run_dir)

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"
    logging.info(f"Creating server (host: {hostname}, ip: {local_ip})")

    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "libero", "ckpt": ckpt_path},
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="starVLA training run dir (contains config.yaml + dataset_statistics.json)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Defaults to <run_dir>/final_model/pytorch_model.pt")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_wait_ms", type=int, default=20)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_argparser()
    main(parser.parse_args())
