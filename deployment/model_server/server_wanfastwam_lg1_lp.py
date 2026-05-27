"""WebSocket policy server for LIBERO-plus on lg1 (24 GB 4090).

Wraps `server_wanfastwam_starvla.py`'s main loop with three load-time
overrides that mirror what we do in the in-process worker
(eval_libero_plus_lg1.py); they are required to fit the 5B Wan video DiT +
1B ActionDiT on a 24 GB 4090 and to point text/VAE/DiT weights at lg1
copies rather than the LFT-W02 absolute paths baked into the action ckpt:

  (1) Repoint `framework.world_model.text_embed_cache_path` to the lg1
      pre-computed UMT5 cache covering all 10002 LIBERO-plus task strings.
      Drops the live UMT5 (~11 GB) from GPU memory.
  (2) `framework.world_model.skip_transformer_load = True`: skip the
      diffusers Wan2.2-TI2V-5B/transformer/*.safetensors load. The full
      DiT state is re-installed by the action ckpt's `state_dict` anyway,
      and the 19 GB diffusers shards are NOT on lg1.
  (3) `framework.action_dit.skip_pretrained_load = True`: same trick for
      the ActionDiT Wan-interp warm-start `.pt` — it lives only on
      LFT-W02 and is overridden by the action ckpt.

The Wan2.2 loader runs on CPU (monkey-patched device=`cpu`) then the
whole model is cast to bf16 and moved to cuda once — avoids OOM from
overlapping fp32 ckpt + bf16 model on a 24 GB GPU.

Run:
    python deployment/model_server/server_wanfastwam_lg1_lp.py \
        --run_dir /data3/junjie/fwalign_olabots_ckpts \
        --ckpt_path /data3/junjie/fwalign_olabots_ckpts/final_model/pytorch_model.pt \
        --port 6694 --use_bf16 --batch_size 5 --batch_wait_ms 20 \
        [--joint_faithful]   # for fwjoint ckpt
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
        return
    pp = ds[chosen].get("state", ds[chosen].get("proprio"))
    if pp is None or "min" not in pp or "max" not in pp:
        return
    if not hasattr(model, "set_state_stats"):
        return
    model.set_state_stats(pp["min"], pp["max"])
    logging.info(f"[stats] installed state stats key={chosen} dim={len(pp['min'])}")


def _load_vla_for_lg1(ckpt_path: str, text_cache_dir: str):
    """Like baseframework.from_pretrained but with the 3 lg1 overrides
    above + CPU-first loader. See module docstring for rationale."""
    ckpt_p = Path(ckpt_path)
    model_config, norm_stats = read_mode_config(ckpt_p)

    wm = model_config.get("framework", {}).get("world_model", {})
    if isinstance(wm, dict):
        logging.info(f"[lg1] Repointing text_embed_cache_path to {text_cache_dir} "
                     f"(was {wm.get('text_embed_cache_path')!r})")
        wm["text_embed_cache_path"] = text_cache_dir
        wm["load_text_encoder"] = False
        wm["skip_transformer_load"] = True
        _lg1_base_wm = "/data3/junjie/weights/Wan2.2-TI2V-5B-Diffusers"
        if os.path.exists(os.path.join(_lg1_base_wm, "vae", "config.json")):
            logging.info(f"[lg1] Repointing base_wm to {_lg1_base_wm} (was {wm.get('base_wm')!r})")
            wm["base_wm"] = _lg1_base_wm


    adit = model_config.get("framework", {}).get("action_dit", {})
    if isinstance(adit, dict):
        adit["pretrained_path"] = None
        adit["skip_pretrained_load"] = True

    cfg = dict_to_namespace(model_config)
    cfg.trainer.pretrained_checkpoint = None

    # Force Wan2.2 loader onto CPU to avoid OOM on 24 GB 4090.
    from starVLA.model.modules.world_model import Wan2_fastwam as _wm_module
    _orig_load = _wm_module.load_wan22_ti2v_5b_components

    def _patched_load(*args, **kwargs):
        kwargs["device"] = "cpu"
        return _orig_load(*args, **kwargs)

    _wm_module.load_wan22_ti2v_5b_components = _patched_load
    try:
        model = build_framework(cfg=cfg)
    finally:
        _wm_module.load_wan22_ti2v_5b_components = _orig_load
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
    logging.info(f"Loading framework from {args.ckpt_path}")
    vla = _load_vla_for_lg1(args.ckpt_path, args.text_cache_dir)

    _install_state_stats(vla, args.run_dir)

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
    torch.cuda.empty_cache()

    # If we're serving a joint-faithful ckpt, monkey-patch predict_action to
    # dispatch to predict_action_joint so the client doesn't need to know.
    if args.joint_faithful:
        if not hasattr(vla, "predict_action_joint"):
            raise RuntimeError("--joint_faithful set but model has no predict_action_joint")
        logging.info("[lg1] joint_faithful=True → rebinding predict_action → predict_action_joint")
        vla.predict_action = vla.predict_action_joint  # type: ignore[assignment]

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
        metadata={"env": "libero_plus", "ckpt": args.ckpt_path,
                  "joint_faithful": bool(args.joint_faithful)},
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt_path", default=None,
                        help="defaults to <run_dir>/final_model/pytorch_model.pt")
    parser.add_argument("--text_cache_dir", required=True,
                        help="precomputed UMT5 text cache covering all LIBERO-plus tasks")
    parser.add_argument("--port", type=int, default=6694)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batch_wait_ms", type=int, default=20)
    parser.add_argument("--joint_faithful", action="store_true",
                        help="dispatch predict_action_joint (use for fwjoint ckpt)")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argparser().parse_args()
    if not args.ckpt_path:
        args.ckpt_path = os.path.join(args.run_dir, "final_model", "pytorch_model.pt")
    main(args)
