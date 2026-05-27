"""LIBERO-plus eval client (websocket → starVLA model server).

Forked from examples/LIBERO/eval_files/eval_libero.py to (a) walk LIBERO-plus
benchmark (2400+ tasks/suite instead of 10), (b) tag each task with the
disturbance category from `libero/libero/benchmark/task_classification.json`,
(c) skip per-episode video writes (would be 10k+ mp4s), (d) emit a richer
per-worker JSON with disturbance breakdown so the orchestrator can aggregate.

Talks to the server via the websocket `ModelClient` — identical API to the
standard LIBERO client, so the lg1 server (deployment/model_server/
server_wanfastwam_lg1_lp.py) Just Works.
"""

import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from typing import Optional

import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    return np.asarray([1.0 - 2.0 * (v > 0.5)], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 6694
    resize_size = [224, 224]
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 30
    num_trials_per_task: int = 1
    task_id_start: int = 0
    task_id_end: int = -1
    worker_id: int = 0
    video_out_path: str = "experiments/libero_plus/logs"
    seed: int = 7
    pretrained_path: str = ""  # passed to ModelClient for action norm stats
    save_video: bool = False


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _suite_max_steps(suite: str) -> int:
    return {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }[suite]


def _load_disturbance_map(libero_home: str, suite: str) -> dict:
    """Return {1-indexed task_id: (category, name)} from LIBERO-plus's
    `libero/libero/benchmark/task_classification.json`. Empty dict if the
    file is missing (degrades gracefully to plain SR)."""
    path = os.path.join(libero_home, "libero/libero/benchmark/task_classification.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        mapping = json.load(f).get(suite, [])
    return {item["id"]: (item["category"], item["name"]) for item in mapping}


def eval_libero(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"Args: {json.dumps(dataclasses.asdict(args), indent=2)}")

    np.random.seed(args.seed)

    bd = benchmark.get_benchmark_dict()
    task_suite = bd[args.task_suite_name]()
    n_tasks = task_suite.n_tasks
    end = n_tasks if args.task_id_end < 0 else min(args.task_id_end, n_tasks)
    start = max(0, args.task_id_start)
    logging.info(f"Suite {args.task_suite_name} n_tasks={n_tasks}; slice [{start}, {end})")

    max_steps = _suite_max_steps(args.task_suite_name)

    libero_home = os.environ.get("LIBERO_HOME", "")
    id2cat = _load_disturbance_map(libero_home, args.task_suite_name)

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    total_episodes = total_successes = 0
    per_task: dict = {}
    disturb_res: dict = {}
    if id2cat:
        for cat, _ in id2cat.values():
            disturb_res.setdefault(cat, {"total_count": 0, "success_count": 0})

    for task_id in tqdm.tqdm(range(start, end)):
        cat, name = id2cat.get(task_id + 1, (None, None))
        env = None
        task_episodes = task_successes = 0
        task_description = ""
        try:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            for ep in range(args.num_trials_per_task):
                client_model.reset(task_description=task_description)
                env.reset()
                obs = env.set_init_state(initial_states[ep])
                t = step = 0
                done = False
                replay = []
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    if args.save_video:
                        replay.append(img)
                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))
                    ex = {
                        "image": [img, wrist],
                        "lang": str(task_description),
                        "state": state,
                    }
                    resp = client_model.step(example=ex, step=step)
                    ra = resp["raw_action"]
                    wv = np.asarray(ra["world_vector"], dtype=np.float32).reshape(-1)
                    rot = np.asarray(ra["rotation_delta"], dtype=np.float32).reshape(-1)
                    grip = _binarize_gripper_open(np.asarray(ra["open_gripper"], dtype=np.float32))
                    act = np.concatenate([wv, rot, grip], axis=0)
                    obs, _, done, _ = env.step(act.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        if cat:
                            disturb_res[cat]["success_count"] += 1
                        break
                    t += 1
                    step += 1
                task_episodes += 1
                total_episodes += 1
                if cat:
                    disturb_res[cat]["total_count"] += 1

                if args.save_video and replay:
                    import imageio
                    suffix = "success" if done else "failure"
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) /
                        f"rollout_task{task_id}_ep{ep}_{suffix}.mp4",
                        [np.asarray(x) for x in replay], fps=20,
                    )
        except Exception as e:
            # A single task (e.g. a disturbance renderer edge case) must never
            # kill the whole worker — log it, count remaining trials as failures
            # so SR stays honest, and move on. This is the "no eval errors" req.
            import traceback
            logging.error(f"[worker {args.worker_id}] task {task_id} ({name}) raised "
                          f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            for _ in range(max(0, args.num_trials_per_task - task_episodes)):
                task_episodes += 1
                total_episodes += 1
                if cat:
                    disturb_res[cat]["total_count"] += 1
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        per_task[task_id] = {
            "episodes": task_episodes,
            "successes": task_successes,
            "description": task_description,
            "category": cat,
            "name": name,
        }
        if (task_id - start + 1) % 25 == 0:
            sr = total_successes / max(1, total_episodes)
            logging.info(f"worker={args.worker_id} progress {task_id-start+1}/{end-start} "
                         f"SR={sr*100:.2f}% ({total_successes}/{total_episodes})")

    summary = {
        "task_suite_name": args.task_suite_name,
        "task_id_start": start,
        "task_id_end": end,
        "worker_id": args.worker_id,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "rate": total_successes / total_episodes if total_episodes else 0.0,
        "disturb_breakdown": disturb_res,
        "per_task": per_task,
    }
    summary_path = pathlib.Path(args.video_out_path) / f"_summary_w{args.worker_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"[worker {args.worker_id}] DONE  SR={summary['rate']*100:.2f}%  "
                 f"({total_successes}/{total_episodes})  written {summary_path}")


if __name__ == "__main__":
    tyro.cli(eval_libero)
