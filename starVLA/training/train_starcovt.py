# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Training entry for QwenCoVT (visual Chain-of-Thought, segmentation-only).

Reuses the ``VLAMTrainer`` infrastructure from ``train_starvlm.py`` but:
    - builds the dataloader **after** the framework so that the dataset
      uses the framework's tokenizer (with CoVT special tokens registered);
    - dispatches each batch through ``model.forward_vlm(batch)``, which
      returns a ``dict`` of loss components ({"vlm_loss", "seg_loss"}).

Usage::

    accelerate launch starVLA/training/train_starcovt.py \
        --config_yaml examples/CoVT/train_files/starvla_covt_qwen3vl.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from starVLA.dataloader.robointer_covt import build_robointer_covt_loader
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.train_starvlm import (
    VLAMTrainer,
    accelerator,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

logger = get_logger(__name__)


class CoVTTrainer(VLAMTrainer):
    """VLAMTrainer override that routes batches through ``forward_vlm``."""

    def _train_step(self, batch_vlm):
        log_dict = {}
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unwrapped = self.accelerator.unwrap_model(self.model)
                losses = unwrapped.forward_vlm(batch_vlm)
                # Apply optional per-key loss scales from cfg.trainer.loss_scale.
                scale_cfg = getattr(self.config.trainer, "loss_scale", {}) or {}
                total = sum(
                    v * float(scale_cfg.get(k, 1.0))
                    for k, v in losses.items()
                    if isinstance(v, torch.Tensor)
                )
            self.accelerator.backward(total)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    log_dict[k] = float(v.detach())
            log_dict["total_loss"] = float(total.detach())
        return log_dict


def prepare_data_with_processor(cfg, model):
    logger.info(
        f"Creating CoVT dataset from `{cfg.datasets.vlm_data.lmdb_path}`"
    )
    processor = model.qwen_vl_interface.processor
    return build_robointer_covt_loader(cfg, processor)


def main(cfg) -> None:
    cfg = wrap_config(cfg)
    setup_directories(cfg=cfg)
    model = build_framework(cfg)
    loader = prepare_data_with_processor(cfg, model)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)

    trainer = CoVTTrainer(
        cfg=cfg,
        model=model,
        vlm_train_dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()

    logger.info("CoVT training complete.")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/CoVT/train_files/starvla_covt_qwen3vl.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clipargs)))
    cfg.config_yaml = args.config_yaml

    main(cfg)
