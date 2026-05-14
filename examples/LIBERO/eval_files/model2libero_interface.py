from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from starVLA.model.tools import read_mode_config


class FastWAMActionEnsembler:
    """FastWAM-style action ensembler. Each replan adds a chunk indexed by absolute
    env timestamp; per-step query averages all predictions targeting that ts.
    Identical to upstream `experiments/libero/action_ensembler.py`.
    """

    def __init__(self):
        self.action_cache: dict[int, list[np.ndarray]] = defaultdict(list)

    def reset(self):
        self.action_cache.clear()

    def add_actions(self, action_chunk: np.ndarray, start_timestamp: int):
        if action_chunk.ndim == 3:
            action_chunk = action_chunk.squeeze(0)
        horizon, action_dim = action_chunk.shape
        for i in range(horizon):
            target_ts = start_timestamp + i
            self.action_cache[target_ts].append(action_chunk[i, :])

    def get_action(self, timestamp: int) -> np.ndarray:
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}")
        preds = self.action_cache[timestamp]
        stacked_preds = np.stack(preds, axis=0)
        return np.mean(stacked_preds, axis=0)

    def cleanup(self, current_timestamp: int):
        keys_to_delete = [ts for ts in self.action_cache.keys() if ts < current_timestamp]
        for ts in keys_to_delete:
            del self.action_cache[ts]


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=True,
        action_ensemble_horizon: Optional[int] = 3,  # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
    ) -> None:

        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)

        # FastWAM-aligned eval: replan every `replan_steps` env steps (vs old open-loop
        # 32-step chunk execution) + average overlapping predictions via ActionEnsembler.
        # Open-loop drift was the dominant SR drop on long-horizon suites (libero_10
        # was -27% vs paper). FastWAM defaults: replan_steps=5, use_action_ensembler=True.
        import os as _os
        self.replan_steps = int(_os.environ.get("REPLAN_STEPS", "5"))
        self.use_fwam_ensembler = bool(int(_os.environ.get("USE_FWAM_ENSEMBLER", "1")))
        if self.use_fwam_ensembler:
            self.fwam_ensembler = FastWAMActionEnsembler()
        else:
            self.fwam_ensembler = None

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        if self.fwam_ensembler is not None:
            self.fwam_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, example: dict, step: int = 0, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # list of images for history

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        # FastWAM-aligned: replan every `replan_steps` (default 5) instead of every
        # `action_chunk_size` (32). Open-loop 32-step execution caused -27% SR drop
        # on libero_10 vs paper. Each replan pushes the predicted chunk into the
        # ensembler indexed by absolute env timestamp, and per-step query averages
        # all predictions targeting that timestamp.
        if step % self.replan_steps == 0:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # [B, chunk, D]
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            chunk_raw = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )
            if self.fwam_ensembler is not None:
                self.fwam_ensembler.add_actions(chunk_raw, start_timestamp=step)
            else:
                self.raw_actions = chunk_raw

        if self.fwam_ensembler is not None:
            current = self.fwam_ensembler.get_action(step)
            self.fwam_ensembler.cleanup(step)
            raw_actions = current[None]
        else:
            # FastWAM-style execute-then-replan: each replan_steps cycle, use
            # the FIRST `replan_steps` actions of the freshly predicted chunk
            # (discard the rest). Index is `step % replan_steps` not
            # `step % action_chunk_size` — using the chunk-size modulo here
            # caused 0% SR in v3-par2 because replan at step=N reset the chunk
            # but we kept indexing at offset N % 32 instead of 0.
            offset = step % self.replan_steps
            raw_actions = self.raw_actions[offset][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """Linear min/max unnorm. Removed the prior `gripper < 0.5 → 0/1` pre-
        binarize which mishandled normalized [-1, 1] gripper values (threshold
        0.5 in [-1, 1] space is biased toward close). The downstream
        `_binarize_gripper_open(v)` in eval_libero.py already binarizes the
        unnormalized gripper at 0.5 in [0, 1] space, which is the correct
        midpoint after the linear unnorm.
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
