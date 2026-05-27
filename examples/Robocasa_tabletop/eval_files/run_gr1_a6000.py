"""Robust Python orchestrator for Robocasa GR1 eval on local A6000 (2 GPU).

Replaces the bash launcher whose `set -e` / EXIT-trap / process-group SIGHUP
kept tearing the whole run down the instant the wait loop hiccuped (servers
survived loading + all 24 clients connected, then the launcher exited and
SIGHUP-killed everything). Python subprocess with start_new_session=True puts
each child in its own session so nothing cascades, and we wait/poll explicitly.

  - 2 model servers (starVLA env), 1 per A6000 GPU, ports 6398/6399.
  - 24 sim clients (robocasa_starVLA env), one per GR1 PnP env, osmesa CPU
    render (clients never touch the GPU -> no EGL/CUDA contention with servers),
    round-robin across the 2 servers.
  - Servers are killed only at the very end (after all clients finish).

Run:
    /data/LFT-W02_data/.conda/envs/starVLA/bin/python \
        examples/Robocasa_tabletop/eval_files/run_gr1_a6000.py
"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

STARVLA_DIR = "/data/LFT-W02_data/junjie/VLA_WM/starVLA"
STARVLA_PY = "/data/LFT-W02_data/.conda/envs/starVLA/bin/python"
ROBOCASA_PY = "/data/LFT-W02_data/.conda/envs/robocasa_starVLA/bin/python"
CKPT = os.environ.get("CKPT", "/data/LFT-W02_data/junjie/VLA_WM/gr1_ola_ckpts/final_model/pytorch_model.pt")
RUN_DIR = str(Path(CKPT).parent.parent)
BASE_WM = "/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers"
DIFFSYNTH_BASE = "/data/LFT-W02_data/junjie/weights"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
    f"/data/LFT-W02_data/junjie/VLA_WM/eval_runs/gr1_a6000_{time.strftime('%Y%m%d_%H%M%S')}")

NUM_GPUS = int(os.environ.get("NUM_GPUS", "2"))
N_EPISODES = int(os.environ.get("N_EPISODES", "50"))
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", "720"))
N_ACTION_STEPS = int(os.environ.get("N_ACTION_STEPS", "12"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
BATCH_WAIT_MS = int(os.environ.get("BATCH_WAIT_MS", "20"))
BASE_PORT = int(os.environ.get("BASE_PORT", "6398"))
HOST = "127.0.0.1"

ENV_NAMES = [
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((HOST, port)) == 0


def main():
    os.makedirs(f"{OUTPUT_DIR}/server_logs", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/client_logs", exist_ok=True)
    log(f"OUTPUT_DIR={OUTPUT_DIR}  CKPT={CKPT}")
    log(f"NUM_GPUS={NUM_GPUS} N_EPISODES={N_EPISODES} BATCH_SIZE={BATCH_SIZE}")

    # 1) Start servers, each in its own session (start_new_session=True) so the
    #    orchestrator dying never SIGHUPs them, and vice-versa.
    servers = []
    for g in range(NUM_GPUS):
        port = BASE_PORT + g
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONPATH"] = STARVLA_DIR
        lf = open(f"{OUTPUT_DIR}/server_logs/server_gpu{g}_p{port}.log", "w")
        p = subprocess.Popen(
            [STARVLA_PY, "deployment/model_server/server_wanfastwam_gr1_a6000.py",
             "--ckpt_path", CKPT, "--run_dir", RUN_DIR,
             "--base_wm", BASE_WM, "--diffsynth_base", DIFFSYNTH_BASE,
             "--port", str(port), "--use_bf16", "--idle_timeout", "-1",
             "--batch_size", str(BATCH_SIZE), "--batch_wait_ms", str(BATCH_WAIT_MS)],
            cwd=STARVLA_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        servers.append((g, port, p))
        log(f"server gpu={g} port={port} pid={p.pid}")

    # 2) Wait for all server ports.
    for g, port, p in servers:
        for _ in range(1800):
            if port_open(port):
                log(f"server :{port} READY")
                break
            if p.poll() is not None:
                log(f"FATAL: server :{port} died during load (rc={p.returncode})")
                _kill_all(servers, [])
                sys.exit(1)
            time.sleep(2)
        else:
            log(f"FATAL: server :{port} timed out")
            _kill_all(servers, [])
            sys.exit(1)

    # 3) Launch all clients (osmesa CPU render; never touch GPU).
    clients = []
    for idx, env_name in enumerate(ENV_NAMES):
        g = idx % NUM_GPUS
        port = BASE_PORT + g
        short = env_name.split("/")[-1].split("_GR1")[0]
        vout = f"{OUTPUT_DIR}/videos/{short}"
        os.makedirs(vout, exist_ok=True)
        cenv = dict(os.environ)
        cenv["CUDA_VISIBLE_DEVICES"] = ""
        cenv["MUJOCO_GL"] = "osmesa"
        cenv["PYOPENGL_PLATFORM"] = "osmesa"
        cenv["PYTHONPATH"] = STARVLA_DIR
        lf = open(f"{OUTPUT_DIR}/client_logs/{short}.log", "w")
        p = subprocess.Popen(
            [ROBOCASA_PY, "examples/Robocasa_tabletop/eval_files/simulation_env.py",
             "--args.host", HOST, "--args.port", str(port),
             "--args.env_name", env_name,
             "--args.n_episodes", str(N_EPISODES),
             "--args.n_envs", "1",
             "--args.max_episode_steps", str(MAX_EPISODE_STEPS),
             "--args.n_action_steps", str(N_ACTION_STEPS),
             "--args.video_out_path", vout,
             "--args.pretrained_path", CKPT],
            cwd=STARVLA_DIR, env=cenv, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        clients.append((short, p))
        time.sleep(1)
    log(f"launched {len(clients)} sim clients")

    # 4) Poll until all clients exit (report progress every ~5 min).
    last_report = time.time()
    while True:
        alive = [s for s, p in clients if p.poll() is None]
        if not alive:
            break
        if time.time() - last_report > 300:
            log(f"{len(alive)}/{len(clients)} clients still running: "
                f"{alive[:4]}{'...' if len(alive) > 4 else ''}")
            last_report = time.time()
        time.sleep(10)
    log("all clients finished")

    # 5) Aggregate per-env SR from client logs.
    results = {}
    for short, _ in clients:
        f = f"{OUTPUT_DIR}/client_logs/{short}.log"
        txt = Path(f).read_text(errors="ignore") if os.path.exists(f) else ""
        m = re.findall(r"[Ss]uccess[_ ]?rate[:=]\s*([0-9.]+)", txt)
        results[short] = float(m[-1]) if m else None
    ok = [v for v in results.values() if v is not None]
    print("=== Robocasa GR1 per-env success rate ===", flush=True)
    for k in sorted(results):
        print(f"  {k}: {results[k]}", flush=True)
    if ok:
        print(f"MEAN over {len(ok)}/{len(results)} envs: {sum(ok)/len(ok):.4f}", flush=True)
    json.dump(results, open(f"{OUTPUT_DIR}/aggregate.json", "w"), indent=2)
    log(f"aggregate.json written to {OUTPUT_DIR}")

    # 6) Cleanup servers.
    _kill_all(servers, clients)


def _kill_all(servers, clients):
    for _, _, p in servers:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    for _, p in clients:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


if __name__ == "__main__":
    main()
