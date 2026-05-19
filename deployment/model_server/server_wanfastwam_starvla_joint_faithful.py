"""Joint-faithful WebSocket server for WanFastWAM ckpts trained with
`framework.fastwam.mot_attention_mode: joint`.

Differs from `server_wanfastwam_starvla.py` only in this:
    after loading the framework, we rebind `predict_action` →
    `predict_action_joint` so the websocket inference handler invokes the
    per-step joint-denoise path (matches `FastWAMJoint.infer_action` upstream).

When to use this vs `server_wanfastwam_starvla.py`:
    - Standard `predict_action` path (the other server): prefill video once,
      reuse video K/V across action denoise steps. Only mathematically faithful
      when the ckpt was trained with the FastWAM mask (A→V = first-frame only).
    - `predict_action_joint` path (this server): re-encode noisy video each
      denoise step (no KV cache reuse). Required for ckpts where action attends
      to the FULL video latent (FastWAMJoint mask). Strictly more expensive.

Same dir layout / env / state-stat install as `server_wanfastwam_starvla.py`.

Usage:
    python deployment/model_server/server_wanfastwam_starvla_joint_faithful.py \
        --run_dir /path/to/joint_trained_run_dir \
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


def _rebind_predict_action_to_joint(model) -> None:
    """Rebind `model.predict_action` → `model.predict_action_joint`.

    The websocket server calls `policy.predict_action(**payload)` unconditionally;
    rather than fork the server protocol or add a per-request flag (which would
    bleed into the eval client too), we swap the attribute on this specific
    instance. The framework class is untouched; only this server's loaded model
    sees the rebind.
    """
    if not hasattr(model, "predict_action_joint"):
        raise AttributeError(
            f"{type(model).__name__} has no `predict_action_joint` method. "
            "Make sure WanFastWAM with the joint-faithful inference branch is installed."
        )
    # Bound method assignment — accessing `model.predict_action_joint` returns
    # an already-bound MethodType, so we can store it under the standard name.
    model.predict_action = model.predict_action_joint
    logging.info("[joint-faithful] rebound predict_action → predict_action_joint on this instance")


def _warn_if_mode_not_joint(model) -> None:
    mode = getattr(model, "mot_attention_mode", None)
    if mode is None:
        logging.warning("[joint-faithful] model has no `mot_attention_mode` attr; "
                        "this server expects WanFastWAM.")
        return
    if mode != "joint":
        logging.warning(
            "[joint-faithful] mot_attention_mode=%r (expected 'joint'). "
            "The inference path will use the full A→V mask anyway, but if the ckpt "
            "was trained with the FastWAM mask the action expert will see a different "
            "attention pattern than at training time → expect SR degradation.",
            mode,
        )
    else:
        logging.info("[joint-faithful] mot_attention_mode=joint  (matches ckpt training)")


def main(args) -> None:
    ckpt_path = args.ckpt_path or os.path.join(args.run_dir, "final_model", "pytorch_model.pt")
    logging.info(f"Loading framework from {ckpt_path}")
    vla = baseframework.from_pretrained(ckpt_path)

    _install_state_stats(vla, args.run_dir)
    _warn_if_mode_not_joint(vla)
    _rebind_predict_action_to_joint(vla)

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
        metadata={
            "env": "libero",
            "ckpt": ckpt_path,
            "inference_path": "joint_faithful",
            "mot_attention_mode": getattr(vla, "mot_attention_mode", "unknown"),
        },
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    logging.info("joint-faithful server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="starVLA training run dir (config.yaml + dataset_statistics.json)")
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
