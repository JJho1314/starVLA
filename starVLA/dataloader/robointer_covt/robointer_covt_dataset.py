"""RoboInter-Data → CoVT-style supervised dataset.

Each sample =
    (primary-camera frame, instruction, GT object-mask) →
    user prompt + assistant response containing
        <think> ... <|anchor_start|><|sam_pad|>×K<|anchor_end|> ... </think>
        <answer> primitive_skill: ...  subtask: ... </answer>

We treat one randomly-sampled frame per episode per epoch.

Notes:
    - Annotations live in an LMDB; per-frame mask in a separate ``sam_mask`` NPZ.
    - The processor (Qwen3-VL ``AutoProcessor``) is owned by the framework and
      passed in so its tokenizer (with special tokens registered by the
      framework) is the source of truth.
"""

from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = -100

ANCHOR_START_TOKEN = "<|anchor_start|>"
ANCHOR_END_TOKEN = "<|anchor_end|>"
SAM_PAD_TOKEN = "<|sam_pad|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

SYSTEM_MESSAGE = "You are a helpful robotics assistant."
USER_PROMPT = (
    "Given this robot view and the task instruction \"{instr}\", "
    "identify the relevant object you need to manipulate, "
    "then output the primitive skill and the current subtask."
)


def _build_cot_response(skill: str, subtask: str, num_sam_tokens: int) -> str:
    """Construct the assistant response with CoT visual anchors."""
    sam_pad = ANCHOR_START_TOKEN + SAM_PAD_TOKEN * num_sam_tokens + ANCHOR_END_TOKEN
    cot = f"The relevant object segmentation is {sam_pad}. "
    answer = f"primitive_skill: {skill}. subtask: {subtask}"
    return f"{THINK_OPEN}{cot}{THINK_CLOSE}{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"


def _build_plain_response(skill: str, subtask: str) -> str:
    """Warmup-stage response (no anchor token)."""
    return f"{ANSWER_OPEN}primitive_skill: {skill}. subtask: {subtask}{ANSWER_CLOSE}"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class RoboInterCoVTConfig:
    """All paths/knobs for the RoboInter CoVT dataset."""
    lmdb_path: str
    video_dir: str
    sam_mask_roots: Sequence[str]   # list of dirs that contain ``<episode>.npz``
    num_sam_tokens: int = 8
    include_anchor_in_prompt: bool = True
    image_size: int = 256
    max_seq_len: int = 1024
    num_workers: int = 4


class RoboInterCoVTDataset(Dataset):
    def __init__(
        self,
        cfg: RoboInterCoVTConfig,
        processor,                    # transformers AutoProcessor (Qwen3-VL)
    ) -> None:
        self.cfg = cfg
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # Index sam_mask files by episode key.
        self._sam_index = self._build_sam_index(cfg.sam_mask_roots)

        # Open LMDB read-only and keep only episode keys we have masks for.
        self._lmdb_path = cfg.lmdb_path
        self._lmdb = None  # opened lazily per worker
        env = lmdb.open(cfg.lmdb_path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            all_keys = [k.decode() for k, _ in txn.cursor()]
        env.close()
        self.episode_keys = [k for k in all_keys if k in self._sam_index]
        if not self.episode_keys:
            raise RuntimeError(
                f"No episode in LMDB `{cfg.lmdb_path}` has a matching sam_mask file. "
                "Check `sam_mask_roots`."
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_sam_index(roots: Sequence[str]) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, files in os.walk(root):
                for fname in files:
                    if fname.endswith(".npz"):
                        index[fname[:-4]] = os.path.join(dirpath, fname)
        return index

    def _ensure_lmdb(self) -> "lmdb.Environment":
        if self._lmdb is None:
            self._lmdb = lmdb.open(
                self._lmdb_path, readonly=True, lock=False, readahead=False
            )
        return self._lmdb

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.episode_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.episode_keys[idx]
        with self._ensure_lmdb().begin() as txn:
            ep = pickle.loads(txn.get(key.encode()))

        frame_id = self._sample_frame(ep)
        ann = ep[frame_id]

        instr = ann.get("instruction_add") or "manipulate the object"
        skill = ann.get("primitive_skill") or "act"
        subtask = ann.get("substask") or "perform the task"

        # Load video frame + mask.
        image_pil = self._load_frame(key, frame_id)
        gt_mask = self._load_mask(key, frame_id)            # [1, H, W] float in {0, 1}

        # Build conversation.
        if self.cfg.include_anchor_in_prompt:
            response = _build_cot_response(skill, subtask, self.cfg.num_sam_tokens)
        else:
            response = _build_plain_response(skill, subtask)

        sample = self._tokenize(image_pil, instr, response)
        sample["gt_masks"] = torch.from_numpy(gt_mask).float()  # [1, H, W]
        sample["images_pil"] = image_pil
        return sample

    # ------------------------------------------------------------------ #
    def _sample_frame(self, ep: Dict[int, dict]) -> int:
        """Pick a random frame inside the first valid time_clip segment."""
        any_frame = next(iter(ep.values()))
        time_clip = any_frame.get("time_clip")
        if time_clip:
            s, e = time_clip[0]
            valid = [f for f in ep.keys() if s <= f < e]
            if valid:
                return random.choice(valid)
        return random.choice(list(ep.keys()))

    def _load_frame(self, key: str, frame_id: int) -> Image.Image:
        path = os.path.join(self.cfg.video_dir, f"{key}.mp4")
        vr = VideoReader(path, ctx=cpu(0))
        frame_id = min(frame_id, len(vr) - 1)
        arr = vr[frame_id].asnumpy()
        img = Image.fromarray(arr)
        return img.resize((self.cfg.image_size, self.cfg.image_size))

    def _load_mask(self, key: str, frame_id: int) -> np.ndarray:
        npz_path = self._sam_index[key]
        with np.load(npz_path, allow_pickle=True) as z:
            masks = z["masks"]  # [N_obj, T, 1, H, W]
        frame_id = min(frame_id, masks.shape[1] - 1)
        m = masks[0, frame_id, 0].astype(np.float32)        # [H, W]
        return m[None, ...]                                 # [1, H, W]

    # ------------------------------------------------------------------ #
    def _tokenize(self, image: Image.Image, instr: str, response: str) -> Dict[str, torch.Tensor]:
        """Apply Qwen3-VL chat template, producing input_ids/attention_mask/labels.

        Strategy: we tokenize twice — once with assistant content, once
        without (add_generation_prompt=True) — and use the length difference
        to construct labels that mask everything before the response.
        """
        user_msg = {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT.format(instr=instr)},
        ]}
        sys_msg = {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]}
        asst_msg = {"role": "assistant", "content": [{"type": "text", "text": response}]}

        # Length of prompt-only (everything to be masked).
        prompt_only = self.processor.apply_chat_template(
            [[sys_msg, user_msg]],
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=False,
        )
        prompt_len = int(prompt_only["input_ids"].shape[1])

        full = self.processor.apply_chat_template(
            [[sys_msg, user_msg, asst_msg]],
            tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", padding=False,
        )
        input_ids = full["input_ids"][0]
        attn_mask = full["attention_mask"][0]
        # Truncate if needed.
        max_len = self.cfg.max_seq_len
        if input_ids.size(0) > max_len:
            input_ids = input_ids[:max_len]
            attn_mask = attn_mask[:max_len]

        labels = input_ids.clone()
        labels[: min(prompt_len, labels.size(0))] = IGNORE_INDEX

        out = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }
        # Pixel/grid tensors are batch-uniform per Qwen3-VL — pass through.
        for k in ("pixel_values", "image_grid_thw"):
            if k in full:
                out[k] = full[k][0] if full[k].dim() == 2 else full[k].squeeze(0)
        return out


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #
@dataclass
class CoVTCollator:
    pad_token_id: int
    padding_side: str = "right"

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [s["input_ids"] for s in samples]
        attn = [s["attention_mask"] for s in samples]
        lbl = [s["labels"] for s in samples]

        input_ids = self._pad(ids, self.pad_token_id)
        attention_mask = self._pad(attn, 0)
        labels = self._pad(lbl, IGNORE_INDEX)

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat(
                [s["pixel_values"] for s in samples], dim=0
            ) if "pixel_values" in samples[0] else None,
            "image_grid_thw": torch.stack(
                [s["image_grid_thw"] for s in samples], dim=0
            ) if "image_grid_thw" in samples[0] else None,
            "gt_masks": [s["gt_masks"] for s in samples],
            "images_pil": [s["images_pil"] for s in samples],
        }
        # Strip None entries (Qwen3-VL forward accepts missing keys).
        return {k: v for k, v in batch.items() if v is not None}

    def _pad(self, seqs: List[torch.Tensor], value: int) -> torch.Tensor:
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), value, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            n = s.size(0)
            if self.padding_side == "right":
                out[i, :n] = s
            else:
                out[i, -n:] = s
        return out


# --------------------------------------------------------------------------- #
# Convenience builder used by build_dataloader()
# --------------------------------------------------------------------------- #
def build_robointer_covt_loader(cfg, processor) -> DataLoader:
    """Construct a DataLoader from the OmegaConf ``cfg.datasets.vlm_data`` block.

    Required YAML keys under ``cfg.datasets.vlm_data``:
        lmdb_path, video_dir, sam_mask_roots (list[str]),
        num_sam_tokens, image_size, model_max_length,
        per_device_batch_size, num_workers
    """
    d = cfg.datasets.vlm_data
    ds_cfg = RoboInterCoVTConfig(
        lmdb_path=d.lmdb_path,
        video_dir=d.video_dir,
        sam_mask_roots=list(d.sam_mask_roots),
        num_sam_tokens=int(d.get("num_sam_tokens", 8)),
        include_anchor_in_prompt=bool(d.get("include_anchor_in_prompt", True)),
        image_size=int(d.get("image_size", 256)),
        max_seq_len=int(d.get("model_max_length", 1024)),
        num_workers=int(d.get("num_workers", 4)),
    )
    dataset = RoboInterCoVTDataset(ds_cfg, processor)
    pad_id = processor.tokenizer.pad_token_id or 0
    side = getattr(processor.tokenizer, "padding_side", "right")
    return DataLoader(
        dataset,
        batch_size=int(d.per_device_batch_size),
        shuffle=True,
        num_workers=ds_cfg.num_workers,
        collate_fn=CoVTCollator(pad_token_id=pad_id, padding_side=side),
        pin_memory=True,
    )
