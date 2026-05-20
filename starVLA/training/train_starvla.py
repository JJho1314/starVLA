# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir):
    """Prepare VLA training and (optional) validation dataloaders.

    Returns:
        (train_dataloader, val_dataloader_or_None)

    Validation split is opt-in via either of:
        cfg.datasets.vla_data.val_fraction    (float in (0, 1))
        cfg.datasets.vla_data.val_num_samples (int)

    If neither is set, ``val_dataloader_or_None`` is ``None`` and behavior
    is identical to before this change.
    """
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    full_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    accelerator.dataloader_config.dispatch_batches = False

    train_dl, val_dl = _maybe_split_val_loader(full_train_dataloader, cfg)

    dist.barrier()
    return train_dl, val_dl


def _maybe_split_val_loader(full_loader, cfg):
    """Hold out a deterministic Subset for validation.

    Picks the held-out indices via a numpy permutation seeded with
    ``cfg.seed`` so train/val sets are reproducible across runs.

    Returns the original loader unchanged (and ``None``) when no
    validation budget is configured.
    """
    val_frac_raw = getattr(cfg.datasets.vla_data, "val_fraction", 0.0)
    val_n_raw = getattr(cfg.datasets.vla_data, "val_num_samples", 0)
    val_frac = float(val_frac_raw or 0.0)
    val_n = int(val_n_raw or 0)
    if val_frac <= 0 and val_n <= 0:
        return full_loader, None

    from torch.utils.data import DataLoader, Subset

    ds = full_loader.dataset
    n = len(ds)
    if val_n <= 0:
        val_n = max(1, int(n * val_frac))
    # Cap: never let val eat more than 25% of data.
    val_n = min(val_n, max(1, n // 4))

    seed = int(getattr(cfg, "seed", 42))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:val_n].tolist()
    train_idx = perm[val_n:].tolist()

    def _rebuild(subset, shuffle):
        nw = full_loader.num_workers
        kw = dict(
            batch_size=full_loader.batch_size,
            collate_fn=full_loader.collate_fn,
            num_workers=nw,
            pin_memory=full_loader.pin_memory,
            shuffle=shuffle,
            drop_last=True,
        )
        # `persistent_workers` / `prefetch_factor` are only valid when num_workers>0.
        if nw > 0:
            kw["persistent_workers"] = getattr(full_loader, "persistent_workers", False)
            pf = getattr(full_loader, "prefetch_factor", None)
            if pf is not None:
                kw["prefetch_factor"] = pf
        return DataLoader(subset, **kw)

    train_dl = _rebuild(Subset(ds, train_idx), shuffle=True)
    val_dl = _rebuild(Subset(ds, val_idx), shuffle=False)
    logger.info(
        f"VLA val split: held out {len(val_idx)} / {n} samples "
        f"({len(val_idx) / n * 100:.2f}%); seed={seed}."
    )
    return train_dl, val_dl


def _derive_max_steps_from_epochs(cfg, vla_train_dataloader, accelerator) -> None:
    """If `cfg.trainer.num_epochs` is set (>0), derive `cfg.trainer.max_train_steps`
    from it (FastWAM-style). Otherwise leave `max_train_steps` unchanged.

    optimizer steps per epoch = len(dataloader) / (world_size × grad_accum)
    Note: at this point the dataloader has not been `accelerator.prepare`-d yet,
    so `len(dataloader)` is the unsharded count over the global dataset.
    """
    raw = getattr(cfg.trainer, "num_epochs", None)
    try:
        num_epochs = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        num_epochs = None
    if num_epochs is None or num_epochs <= 0:
        return

    world_size = max(1, int(getattr(accelerator, "num_processes", 1)))
    grad_accum = max(1, int(getattr(cfg.trainer, "gradient_accumulation_steps", 1)))
    n_batches = len(vla_train_dataloader)
    steps_per_epoch = max(1, n_batches // (world_size * grad_accum))
    derived = int(round(num_epochs * steps_per_epoch))

    old = getattr(cfg.trainer, "max_train_steps", None)
    cfg.trainer.max_train_steps = derived
    logger.info(
        f"[epochs] num_epochs={num_epochs} world_size={world_size} grad_accum={grad_accum} "
        f"len(dataloader)={n_batches} -> steps_per_epoch={steps_per_epoch} "
        f"-> max_train_steps={derived} (was {old})"
    )


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_val_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_val_dataloader = vla_val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        if self.vla_val_dataloader is not None:
            (
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vla_val_dataloader,
            ) = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vla_val_dataloader,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # When DeepSpeed bypasses `accelerator.accumulate`, sync_gradients
            # stays True every micro-batch — use the manual flag set inside
            # `_train_step` instead, falling back to accelerator for non-DS.
            sync = getattr(self, "_last_was_sync_step", None)
            if sync is None:
                sync = self.accelerator.sync_gradients

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            # All per-optimizer-step bookkeeping (counter increment, eval, log,
            # save, max-steps check) must be gated by `sync`. Otherwise, with DS
            # grad_accum > 1 and `completed_steps == 0`, eval/log fire every
            # micro-batch (since `0 % anything == 0`) and the loop livelocks.
            if sync:
                # Step the LR scheduler exactly once per optimizer update so the
                # cosine schedule covers `max_train_steps` optimizer steps, not
                # `grad_accum * max_train_steps` micro-batches.
                self.lr_scheduler.step()
                progress_bar.update(1)
                self.completed_steps += 1

                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)

                # Optional validation pass on the held-out split.
                val_interval = int(getattr(self.config.trainer, "val_interval", 0) or 0)
                if (
                    val_interval > 0
                    and self.vla_val_dataloader is not None
                    and self.completed_steps % val_interval == 0
                ):
                    step_metrics.update(self._eval_on_validation())

                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

                if self.completed_steps >= self.config.trainer.max_train_steps:
                    break

        self._finalize_training()

    def _eval_on_validation(self) -> dict:
        """Run model.forward over a few validation batches and return averaged
        per-loss metrics with the ``val/`` wandb prefix.

        Activated when ``trainer.val_interval > 0`` and a validation
        dataloader was built in :func:`prepare_data`. Costs one extra
        forward per ``val_num_batches`` batches per ``val_interval`` steps —
        no backward, no optimizer step.
        """
        if self.vla_val_dataloader is None:
            return {}

        num_batches = int(getattr(self.config.trainer, "val_num_batches", 8) or 8)
        was_training = self.model.training
        self.model.eval()

        sums_cpu: dict[str, float] = {}    # CPU floats, easier to gather_object
        n_batches_seen = 0

        val_iter = iter(self.vla_val_dataloader)
        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(self.vla_val_dataloader)
                    try:
                        batch = next(val_iter)
                    except StopIteration:
                        break

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.accelerator.unwrap_model(self.model).forward(batch)

                for k, v in out.items():
                    if not (k.endswith("_loss") and isinstance(v, torch.Tensor)):
                        continue
                    sums_cpu[k] = sums_cpu.get(k, 0.0) + float(v.detach().float().item())
                n_batches_seen += 1

        if was_training:
            self.model.train()

        # Collect per-rank (sums, count) at rank 0 — gather_object handles ranks
        # that saw different loss key sets without deadlocking on collective ops.
        all_sums = self.accelerator.gather_for_metrics([sums_cpu], use_gather_object=True)
        all_counts = self.accelerator.gather_for_metrics(
            [int(n_batches_seen)], use_gather_object=True
        )

        if not self.accelerator.is_main_process:
            return {}

        total_count = int(sum(all_counts))
        if total_count == 0:
            return {}

        merged: dict[str, float] = {}
        for rank_sums in all_sums:
            for k, v in rank_sums.items():
                merged[k] = merged.get(k, 0.0) + float(v)

        out_metrics: dict[str, float] = {
            f"val/{k}": v / total_count for k, v in merged.items()
        }
        if out_metrics:
            out_metrics["val/total_loss"] = float(sum(out_metrics.values()))
        return out_metrics

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

        # Hard-fail if grad_accum disagrees across config/accelerator/DeepSpeed —
        # this is the kind of mismatch that silently halves effective batch and
        # was costly to diagnose. Run on every rank.
        cfg_ga = int(getattr(self.config.trainer, "gradient_accumulation_steps", 1))
        acc_ga = int(self.accelerator.gradient_accumulation_steps)
        if cfg_ga != acc_ga:
            raise RuntimeError(
                f"gradient_accumulation_steps mismatch: trainer config={cfg_ga} "
                f"vs accelerator={acc_ga}. Ensure `--gradient_accumulation_steps {cfg_ga}` "
                f"is passed to `accelerate launch` and ds_config has `auto`."
            )
        try:
            from accelerate.utils import DistributedType
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
                if ds_plugin is not None:
                    ds_ga = int(ds_plugin.deepspeed_config.get("gradient_accumulation_steps", -1))
                    if ds_ga != cfg_ga:
                        raise RuntimeError(
                            f"DeepSpeed gradient_accumulation_steps={ds_ga} != trainer "
                            f"config={cfg_ga}. Check ds_config gradient_accumulation_steps "
                            f"resolves to `auto` and accelerate launch propagates the value."
                        )
        except ImportError:
            pass

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step.

        `accelerator.accumulate(model)` calls `model.no_sync()` for non-sync
        micro-batches, which raises `AssertionError: no_sync context manager
        is incompatible with gradient partitioning logic of ZeRO stage 2`.
        DeepSpeed engine already handles gradient accumulation internally —
        backward accumulates, optimizer.step() is a no-op except on the
        accumulation boundary — so we just skip the accelerate context for DS.

        When we skip `accumulate()`, `accelerator.sync_gradients` is no longer
        toggled, so we maintain our own micro-step counter and expose
        `self._last_was_sync_step` for the train loop to gate `completed_steps`.
        """
        from accelerate.utils import DistributedType
        is_deepspeed = self.accelerator.distributed_type == DistributedType.DEEPSPEED
        if is_deepspeed:
            accumulate_ctx = contextlib.nullcontext()
        else:
            accumulate_ctx = self.accelerator.accumulate(self.model)
        with accumulate_ctx:
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                # Sum every *_loss tensor in the output dict (e.g. "video_loss"
                # for FastWAM-style auxiliary supervision). Frameworks should
                # already apply their own lambda weights before returning.
                total_loss = sum(
                    v for k, v in output_dict.items()
                    if k.endswith("_loss") and isinstance(v, torch.Tensor)
                )

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # NOTE: lr_scheduler.step() is now called from the train loop, gated
            # by the optimizer-boundary sync flag — calling it every micro-batch
            # burns through the cosine schedule grad_accum× too fast.

        if is_deepspeed:
            self._ds_micro_step = getattr(self, "_ds_micro_step", 0) + 1
            grad_accum = max(1, int(self.accelerator.gradient_accumulation_steps))
            self._last_was_sync_step = (self._ds_micro_step % grad_accum == 0)

        metrics = {"action_dit_loss": action_loss.item()}
        for k, v in output_dict.items():
            if k.endswith("_loss") and k != "action_loss" and isinstance(v, torch.Tensor):
                metrics[k] = v.item()
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vla_val_dataloader = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    _derive_max_steps_from_epochs(cfg, vla_train_dataloader, accelerator)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_val_dataloader=vla_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
