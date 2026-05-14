"""LIBERO benchmark — data config, embodiment tags, and mixtures.

Two DataConfig variants for the same LIBERO Franka data:

* ``Libero4in1DataConfig`` (robot_type ``libero_franka``):
    Single-observation + 8-step action chunk, action-only normalization.
    Used by the existing VLM4A / WM4A / cotrain models — DO NOT change
    the schema / delta_indices / normalization here without auditing those
    consumers; this class is the legacy, backwards-compatible default.

* ``Libero4in1FastWAMDataConfig`` (robot_type ``libero_franka_fastwam``):
    FastWAM-aligned: 9 video frames at every-4th-step over a 33-step window
    (``num_frames=33, action_video_freq_ratio=4`` in FastWAM's terms),
    32-step action chunk, single proprio frame, and action+state min/max
    normalization. Uses FastWAM's grouped LeRobot metadata:
    ``state.eef_pose`` (6) + ``state.pad`` (1) + ``state.gripper`` (1) and
    ``action.eef`` (6) + ``action.gripper`` (1).
    Used only by ``starvla_wanfastwam_libero.yaml``.

Both share the same physical Franka data and (in FastWAM-aligned) the same
folders post-download. The two configs differ only in which slice of the
trajectory each loads per sample and which keys the transform normalizes.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# Legacy single-frame, 8-step config — DO NOT CHANGE.
# Used by all existing VLM4A / WM4A / cotrain LIBERO models. Action-only
# normalization (6 dims; gripper is left raw); no state in modality_config.
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            # "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # ignore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# FastWAM-aligned config: 9 video frames + 32-step action chunk +
# proprio loaded + grouped action/state normalization.
# Drop-in only for WanFastWAM (its `_build_backbone_images` knows how to
# consume the per-cam-list-of-T-frames format emitted by `_pack_sample`
# when ``data_cfg.multi_frame_video=true``). DO NOT point other models here.
# ---------------------------------------------------------------------------
class Libero4in1FastWAMDataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_pose",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.eef",
        "action.gripper",
    ]
    language_keys = ["annotation.task"]

    # 9 video frames at stride 4 over a 33-step window (matches FastWAM's
    # ``num_frames=33, action_video_freq_ratio=4``).
    observation_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    # 32 actions = transitions across the 33-step window covered by ``observation_indices``.
    action_indices = list(range(32))
    # Single proprio frame at the current step (FastWAM also uses single proprio).
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            # Language is per-current-step (no need to broadcast over video timesteps).
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        keys = self.action_keys + self.state_keys
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=keys),
            StateActionTransform(
                apply_to=keys,
                normalization_modes={
                    "action.eef": "min_max",
                    "action.gripper": "min_max",
                    "state.eef_pose": "min_max",
                    "state.pad": "min_max",
                    "state.gripper": "min_max",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# Robot type registry
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "libero_franka_fastwam": Libero4in1FastWAMDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
    "libero_franka_fastwam": EmbodimentTag.FRANKA,
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    # Legacy mixture used by all existing VLM4A / WM4A / cotrain LIBERO models.
    # Folder names match the data layout that `_1.0.0_lerobot` HPC3 dir holds
    # — DO NOT change without migrating those data directories.
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    # FastWAM-aligned mixture (9 video frames, 32-step action chunk).
    # Set ``data_mix: libero_all_fastwam`` + ``multi_frame_video: true`` in the
    # WanFastWAM training yaml to use this. Folder names match FastWAM's
    # downloaded LIBERO release (``..._no_noops_lerobot``, no ``_1.0.0_``).
    "libero_all_fastwam": [
        ("libero_object_no_noops_lerobot", 1.0, "libero_franka_fastwam"),
        ("libero_goal_no_noops_lerobot", 1.0, "libero_franka_fastwam"),
        ("libero_spatial_no_noops_lerobot", 1.0, "libero_franka_fastwam"),
        ("libero_10_no_noops_lerobot", 1.0, "libero_franka_fastwam"),
    ],
}
