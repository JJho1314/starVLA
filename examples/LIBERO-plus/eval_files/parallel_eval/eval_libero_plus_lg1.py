"""LIBERO-plus zeroshot eval worker for lg1 (8x RTX 4090).

Forked from `eval_libero_model.py` with two changes required for
parity with the V8 FastWAM eval pipeline:

  1. PIL BILINEAR resize (instead of cv.INTER_AREA) — V8 alignment fix.
  2. --joint_faithful flag → dispatches `predict_action_joint` instead of
     `predict_action`. Required for ckpts trained with
     `framework.fastwam.mot_attention_mode: joint` (e.g. fwjoint final_model
     which scored 98.3 V8 on standard LIBERO).

CPU-unseeded noise (V8 RNG fix) is already baked into WanFastWAM and does
not need to be patched at the eval-worker level.

Each invocation runs a contiguous task-index slice [start_idx, end_idx)
of one LIBERO-plus suite on a single GPU (selected via
CUDA_VISIBLE_DEVICES at launch). Per-shard disturbance breakdown is
written to `<output_dir>/logs/<suite>/<start>_to_<end>.json`.
"""
import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import draccus
import imageio
import numpy as np
import tqdm
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

from starVLA.model.framework.base_framework import baseframework, build_framework
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config as _share_read_config
from starVLA.model.tools import read_mode_config


def _load_vla_from_ckpt(ckpt_path: str):
    """Like `baseframework.from_pretrained`, but with three lg1-specific
    overrides:

      (1) `text_embed_cache_path = None` -- LIBERO-plus tasks are zeroshot,
          so the training T5 cache does not contain their hashes; force the
          live T5 path.
      (2) `skip_transformer_load = True` -- the video DiT (Wan2.2-TI2V-5B
          diffusers transformer/, 19 GB) is NOT on lg1 and is fully
          overridden by the action ckpt's `state_dict` anyway.
      (3) `action_dit.skip_pretrained_load = True` -- same logic for the
          ActionDiT linear-interp warm-start `.pt`; it lives only on
          LFT-W02 and is overridden by the action ckpt.

    All weights are loaded on CPU first (loader monkey-patched via a
    temporary override) to avoid duplicating fp32 ckpt state and bf16
    model copies on the 24 GB 4090, then cast to bf16 and moved to cuda.
    """
    ckpt_p = Path(ckpt_path)
    model_config, norm_stats = _share_read_config(ckpt_p)

    # (1) Repoint text_embed_cache_path to the pre-computed lg1 cache that
    # covers all 10002 LIBERO-plus task descriptions. Drop the live T5
    # encoder — it's 11 GB in bf16 and won't fit alongside the 5B Wan DiT
    # + 1B ActionDiT on a 24 GB 4090.
    wm = model_config.get("framework", {}).get("world_model", {})
    libero_plus_cache = os.environ.get(
        "LIBERO_PLUS_TEXT_CACHE",
        "/data3/junjie/libero_plus_text_cache_lg1",
    )
    if isinstance(wm, dict):
        print(f"[eval] Repointing text_embed_cache_path to {libero_plus_cache} "
              f"(was {wm.get('text_embed_cache_path')!r})")
        wm["text_embed_cache_path"] = libero_plus_cache
        wm["load_text_encoder"] = False

    # (2) Skip loading the transformer (video DiT) weights from
    # `${base_wm}/transformer/*.safetensors`. The full DiT state lives in
    # the action ckpt and gets restored via `load_state_dict` below, so the
    # initial diffusers load is redundant — and on lg1 we don't have the
    # ~20 GB Wan2.2-TI2V-5B-Diffusers/transformer/ shards.
    if isinstance(wm, dict):
        wm["skip_transformer_load"] = True

    # (3) Same trick for the ActionDiT Wan-interp backbone init — the trained
    # ActionDiT weights are in the action ckpt, so the *.pt warm-start file
    # (`ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`, only on LFT-W02)
    # is not actually required at eval time.
    adit = model_config.get("framework", {}).get("action_dit", {})
    if isinstance(adit, dict):
        adit["pretrained_path"] = None
        adit["skip_pretrained_load"] = True

    cfg = dict_to_namespace(model_config)
    cfg.trainer.pretrained_checkpoint = None

    # Monkey-patch the Wan2.2 loader so it stays on CPU until we've loaded
    # the ckpt and converted to bf16; otherwise build_framework loads ~13 GB
    # of fp32 weights onto cuda first, then load_state_dict layers another
    # 13 GB on top, then `.to(bf16)` keeps both versions briefly = OOM on a
    # 24 GB 4090. Wan2_fastwam does `from fastwam... import load_...`, so we
    # monkey-patch the *name* as seen from Wan2_fastwam, not the source module.
    from starVLA.model.modules.world_model import Wan2_fastwam as _wm
    _orig_load = _wm.load_wan22_ti2v_5b_components

    def _patched_load(*args, **kwargs):
        kwargs["device"] = "cpu"
        return _orig_load(*args, **kwargs)

    _wm.load_wan22_ti2v_5b_components = _patched_load
    try:
        model = build_framework(cfg=cfg)
    finally:
        _wm.load_wan22_ti2v_5b_components = _orig_load
    model.norm_stats = norm_stats

    if ckpt_p.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(ckpt_p))
    else:
        state = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    # Drop the cpu state dict before allocating bf16 GPU copies.
    del state
    import gc
    gc.collect()
    return model


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return cur_action


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    return np.asarray([1.0 - 2.0 * (v > 0.5)], dtype=np.float32)


def get_logger(file):
    logger = logging.getLogger(f"eval_lg1_{file}")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@dataclasses.dataclass
class Args:
    pretrained_path: str = ""
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    video_out_path: str = "experiments/libero_plus/logs"
    log_path: str = "experiments/libero_plus/logs"
    seed: int = 7
    use_bf16: bool = True
    start_idx: int = -1
    end_idx: int = -1
    output_dir: str = "./output"
    # V8 / framework dispatch
    joint_faithful: bool = False
    save_video: bool = False
    unnorm_key: str = "franka"
    # Override the ckpt's training-time num_inference_steps (default 10 for
    # FastWAM training). Lower = much faster eval at small SR cost; e.g.
    # 4 matches FastWAM's "fast" inference setting.
    num_inference_steps: int = -1


class PolicyModel:
    """In-process VLA wrapper. Loads checkpoint, runs predict_action(_joint).

    V8 changes vs upstream eval_libero_model.py:
      * Resize uses PIL BILINEAR (matches training preprocess).
      * `joint_faithful=True` dispatches to `predict_action_joint`.
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        action_ensemble: bool = True,
        action_ensemble_horizon: int = 3,
        image_size=(224, 224),
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        use_bf16: bool = True,
        joint_faithful: bool = False,
    ):
        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = tuple(image_size)
        self.joint_faithful = bool(joint_faithful)

        vla = _load_vla_from_ckpt(policy_ckpt_path)
        if use_bf16:
            # Convert dtype while model is still partially on cuda from the
            # loader. Doing this before .to("cuda") would keep fp32 ckpt-state
            # weights and bf16 copies coexisting on GPU and OOM a 24 GB 4090.
            vla = vla.to(torch.bfloat16)
        # Force any remaining fp32/cpu tensors onto cuda in bf16.
        self.vla = vla.to("cuda").eval()
        torch.cuda.empty_cache()

        self.action_norm_stats = self._read_action_stats(policy_ckpt_path, self.unnorm_key)
        self.action_chunk_size = self._read_chunk_size(policy_ckpt_path)

        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensembler = (
            AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)
            if action_ensemble
            else None
        )
        self.task_description: Optional[str] = None
        self.raw_actions = None

    @staticmethod
    def _read_action_stats(ckpt_path, unnorm_key):
        _, norm_stats = read_mode_config(Path(ckpt_path))
        if unnorm_key is None:
            assert len(norm_stats) == 1, f"need explicit unnorm_key, choices: {list(norm_stats.keys())}"
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, f"{unnorm_key} not in {list(norm_stats.keys())}"
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def _read_chunk_size(ckpt_path):
        model_config, _ = read_mode_config(Path(ckpt_path))
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image_pil_bilinear(self, image: np.ndarray) -> np.ndarray:
        """V8 fix: PIL BILINEAR matches training preprocess; cv.INTER_AREA caused -1.8pt on libero_10."""
        if image.shape[:2] == self.image_size:
            return image
        pil = Image.fromarray(image)
        pil = pil.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)
        return np.asarray(pil)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        if self.action_ensembler is not None:
            self.action_ensembler.reset()
        self.raw_actions = None

    def step(self, example: dict, step: int = 0) -> dict:
        if example.get("lang") != self.task_description:
            self.reset(example.get("lang"))

        example["image"] = [self._resize_image_pil_bilinear(img) for img in example["image"]]
        vla_input = {
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        if step % self.action_chunk_size == 0:
            if self.joint_faithful:
                response = self.vla.predict_action_joint(example, **vla_input)
            else:
                response = self.vla.predict_action(example, **vla_input)
            normalized = response["normalized_actions"][0]
            if normalized.shape[1] > 7:
                normalized = normalized[:, -7:]
            self.raw_actions = self._unnormalize(normalized, self.action_norm_stats)

        raw = self.raw_actions[step % self.action_chunk_size][None]
        return {
            "raw_action": {
                "world_vector": np.asarray(raw[0, :3]),
                "rotation_delta": np.asarray(raw[0, 3:6]),
                "open_gripper": np.asarray(raw[0, 6:7]),
            }
        }

    @staticmethod
    def _unnormalize(normalized_actions: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = stats.get("mask", np.ones_like(stats["min"], dtype=bool))
        action_high = np.array(stats["max"])
        action_low = np.array(stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )


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


def _suite_max_steps(suite_name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[suite_name]


@draccus.wrap()
def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.start_idx == -1:
        args.start_idx = 0
        args.end_idx = num_tasks_in_suite
    args.end_idx = min(args.end_idx, num_tasks_in_suite)

    log_dir = os.path.join(args.output_dir, f"logs/{args.task_suite_name}")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.start_idx}_{args.end_idx}.log")
    logger = get_logger(log_file)
    logger.info(f"Args: {json.dumps(dataclasses.asdict(args), indent=2)}")
    logger.info(f"joint_faithful = {args.joint_faithful}")

    video_dir = os.path.join(args.output_dir, args.task_suite_name)
    pathlib.Path(video_dir).mkdir(parents=True, exist_ok=True)

    max_steps = _suite_max_steps(args.task_suite_name)
    client_model = PolicyModel(
        policy_ckpt_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        use_bf16=args.use_bf16,
        joint_faithful=args.joint_faithful,
        num_ddim_steps=args.num_inference_steps if args.num_inference_steps > 0 else 10,
    )

    LIBERO_HOME = os.environ.get("LIBERO_HOME")
    assert LIBERO_HOME, "LIBERO_HOME env var must point to LIBERO-plus repo root"
    with open(os.path.join(LIBERO_HOME, "libero/libero/benchmark/task_classification.json")) as f:
        TASK_MAPPING = json.load(f)[args.task_suite_name]

    ID2CATEGORY = {}
    disturb_res: dict = {}
    for item in TASK_MAPPING:
        ID2CATEGORY[item["id"]] = (item["category"], item["name"])
        disturb_res.setdefault(item["category"], {"total_count": 0, "success_count": 0})

    total_episodes, total_successes = 0, 0
    logger.info(
        f"Suite {args.task_suite_name} has {num_tasks_in_suite} tasks; "
        f"processing slice [{args.start_idx}, {args.end_idx})"
    )

    for task_id in tqdm.tqdm(range(args.start_idx, args.end_idx)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in range(args.num_trials_per_task):
            client_model.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            step = 0
            done = False
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if args.save_video:
                    replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                example_dict = {
                    "image": [np.expand_dims(img, 0)[0], np.expand_dims(wrist_img, 0)[0]],
                    "lang": str(task_description),
                }

                response = client_model.step(example=example_dict, step=step)
                raw = response["raw_action"]
                world = np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)
                rot = np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)
                grip = _binarize_gripper_open(np.asarray(raw["open_gripper"], dtype=np.float32))
                delta_action = np.concatenate([world, rot, grip], axis=0)

                obs, _, done, _ = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    disturb_res[ID2CATEGORY[task_id + 1][0]]["success_count"] += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            disturb_res[ID2CATEGORY[task_id + 1][0]]["total_count"] += 1

            if args.save_video and replay_images:
                suffix = "success" if done else "failure"
                imageio.mimwrite(
                    pathlib.Path(video_dir) / f"rollout_{ID2CATEGORY[task_id+1][1]}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=25,
                )

            logger.info(
                f"task={task_id} ep={episode_idx} done={done} "
                f"total {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.1f}%)"
            )

    # Save per-shard disturbance breakdown
    out_json = os.path.join(log_dir, f"{args.start_idx}_to_{args.end_idx}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "suite": args.task_suite_name,
                "start_idx": args.start_idx,
                "end_idx": args.end_idx,
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "success_rate": (total_successes / total_episodes) if total_episodes else 0.0,
                "disturb_breakdown": disturb_res,
                "joint_faithful": args.joint_faithful,
                "pretrained_path": args.pretrained_path,
            },
            f,
            indent=2,
        )
    logger.info(
        f"DONE shard [{args.start_idx},{args.end_idx}): "
        f"{total_successes}/{total_episodes} = {(total_successes/total_episodes)*100 if total_episodes else 0:.2f}%"
    )


if __name__ == "__main__":
    eval_libero()
