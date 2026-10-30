"""Generic training engine for FM-Bench."""

from __future__ import annotations

import json
import math
import os
import signal
from contextlib import nullcontext
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from cardiac_ssl.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


@dataclass
class TrainerConfig:
    """Configuration shared by all FM-Bench SSL methods."""

    epochs: int = 1000
    train_iterations_per_epoch: int | None = 250
    val_iterations_per_epoch: int | None = 50

    precision: str = "bf16"
    gradient_clip_norm: float | None = None

    output_dir: str = "./fmbench_results"
    checkpoint_every: int = 1

    compile_model: bool = False
    find_unused_parameters: bool = False

    stop_on_nonfinite_loss: bool = True

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if (
            self.train_iterations_per_epoch is not None
            and self.train_iterations_per_epoch <= 0
        ):
            raise ValueError(
                "train_iterations_per_epoch must be positive or None"
            )

        if (
            self.val_iterations_per_epoch is not None
            and self.val_iterations_per_epoch <= 0
        ):
            raise ValueError(
                "val_iterations_per_epoch must be positive or None"
            )

        if self.precision not in {
            "fp32",
            "fp16",
            "bf16",
        }:
            raise ValueError(
                "precision must be fp32, fp16, or bf16"
            )

        if self.checkpoint_every <= 0:
            raise ValueError(
                "checkpoint_every must be positive"
            )


class Trainer:
    """Architecture- and SSL-method-independent training engine.

    The model is expected to accept one batch and return either:

        {"loss": scalar_tensor, ...}

    or a scalar loss tensor directly.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable,
        val_loader: Iterable | None,
        config: TrainerConfig,
        scheduler: Any | None = None,
        device: torch.device | str | None = None,
    ):
        self.config = config

        self.is_ddp = (
            dist.is_available()
            and dist.is_initialized()
        )

        self.rank = (
            dist.get_rank()
            if self.is_ddp
            else 0
        )

        self.world_size = (
            dist.get_world_size()
            if self.is_ddp
            else 1
        )

        self.is_main_process = self.rank == 0

        self.device = self._resolve_device(device)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss: float | None = None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        self.output_dir = Path(config.output_dir)

        if self.is_main_process:
            self.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.scaler = self._build_scaler()

        self.exit_training = False

        if hasattr(signal, "SIGUSR1"):
            signal.signal(
                signal.SIGUSR1,
                self._handle_exit_signal,
            )

        if config.compile_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "torch.compile is unavailable"
                )

            self.model = torch.compile(self.model)

        if self.is_ddp:
            if self.device.type == "cuda":
                self.model = DDP(
                    self.model,
                    device_ids=[self.device.index],
                    find_unused_parameters=(
                        config.find_unused_parameters
                    ),
                )
            else:
                self.model = DDP(
                    self.model,
                    find_unused_parameters=(
                        config.find_unused_parameters
                    ),
                )

    def _resolve_device(
        self,
        device: torch.device | str | None,
    ) -> torch.device:
        if self.is_ddp and torch.cuda.is_available():
            local_rank = int(
                os.environ.get("LOCAL_RANK", 0)
            )

            return torch.device(
                "cuda",
                local_rank,
            )

        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda", 0)

        return torch.device("cpu")

    def _build_scaler(self):
        # GradScaler is only needed for fp16.
        if (
            self.device.type == "cuda"
            and self.config.precision == "fp16"
        ):
            return torch.amp.GradScaler("cuda")

        return None

    def _autocast_context(self):
        if self.config.precision == "fp32":
            return nullcontext()

        if self.device.type == "cuda":
            dtype = (
                torch.float16
                if self.config.precision == "fp16"
                else torch.bfloat16
            )

            return torch.autocast(
                device_type="cuda",
                dtype=dtype,
            )

        if (
            self.device.type == "cpu"
            and self.config.precision == "bf16"
        ):
            return torch.autocast(
                device_type="cpu",
                dtype=torch.bfloat16,
            )

        # CPU fp16 is deliberately not enabled.
        return nullcontext()

    def _handle_exit_signal(
        self,
        *args,
    ):
        self.exit_training = True

    @staticmethod
    def _extract_loss(
        output,
    ) -> torch.Tensor:
        if torch.is_tensor(output):
            loss = output

        elif isinstance(output, dict):
            if "loss" not in output:
                raise KeyError(
                    "Model output dictionary must contain 'loss'"
                )

            loss = output["loss"]

        else:
            raise TypeError(
                "Model must return a scalar tensor or "
                "a dictionary containing 'loss'"
            )

        if not torch.is_tensor(loss):
            raise TypeError(
                "loss must be a torch.Tensor"
            )

        if loss.numel() != 1:
            raise ValueError(
                "loss must be scalar"
            )

        return loss

    def _move_to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(
                self.device,
                non_blocking=True,
            )

        if isinstance(obj, dict):
            return {
                key: self._move_to_device(value)
                for key, value in obj.items()
            }

        if isinstance(obj, tuple):
            return tuple(
                self._move_to_device(value)
                for value in obj
            )

        if isinstance(obj, list):
            return [
                self._move_to_device(value)
                for value in obj
            ]

        return obj

    def _forward(self, batch):
        batch = self._move_to_device(batch)

        with self._autocast_context():
            output = self.model(batch)
            loss = self._extract_loss(output)

        return loss, output

    def _reduce_mean(
        self,
        value: float,
    ) -> float:
        tensor = torch.tensor(
            value,
            dtype=torch.float64,
            device=self.device,
        )

        if self.is_ddp:
            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM,
            )
            tensor /= self.world_size

        return float(tensor.item())

    def _iterate_loader(
        self,
        loader: Iterable,
        max_iterations: int | None,
    ):
        iterator = iter(loader)
        iteration = 0

        while (
            max_iterations is None
            or iteration < max_iterations
        ):
            try:
                batch = next(iterator)

            except StopIteration:
                if max_iterations is None:
                    break

                # nnSSL-style fixed-length epochs:
                # restart finite loaders if required.
                iterator = iter(loader)

                try:
                    batch = next(iterator)
                except StopIteration:
                    raise RuntimeError(
                        "Dataloader is empty"
                    )

            yield batch

            iteration += 1

    def train_step(
        self,
        batch,
    ) -> float:
        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss, outputs = self._forward(batch)

        # Keep detached scalar diagnostics exposed by SSL methods.
        # Examples:
        #   SimCLR -> contrastive_accuracy
        #   DINOv2 -> loss_dino, loss_ibot, loss_koleo
        #   V-JEPA -> method-specific scalar components
        #
        # This is deliberately method-agnostic: Trainer never needs
        # to know which SSL objective produced the metrics.
        self.last_train_metrics = {}

        if isinstance(outputs, dict):
            for name, value in outputs.items():
                if name == "loss":
                    continue

                if (
                    isinstance(value, torch.Tensor)
                    and value.numel() == 1
                ):
                    self.last_train_metrics[name] = float(
                        value.detach().float().item()
                    )
                elif isinstance(value, (int, float)):
                    self.last_train_metrics[name] = float(value)

        if (
            self.config.stop_on_nonfinite_loss
            and not torch.isfinite(loss)
        ):
            raise FloatingPointError(
                f"Non-finite training loss: "
                f"{loss.detach().item()}"
            )

        if self.scaler is not None:
            self.scaler.scale(loss).backward()

            if (
                self.config.gradient_clip_norm
                is not None
            ):
                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.scaler.step(
                self.optimizer
            )
            self.scaler.update()

        else:
            loss.backward()

            if (
                self.config.gradient_clip_norm
                is not None
            ):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()

        # Run method-specific updates after the student optimizer step.
        self._after_optimizer_step()

        self.global_step += 1

        return float(
            loss.detach().float().item()
        )

    def _after_optimizer_step(self) -> None:
        """Run an optional method-specific post-optimizer hook."""

        model = self.model

        # Unwrap DistributedDataParallel.
        if isinstance(
            model,
            torch.nn.parallel.DistributedDataParallel,
        ):
            model = model.module

        # Unwrap torch.compile when present.
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        hook = getattr(
            model,
            "on_after_optimizer_step",
            None,
        )

        if hook is not None:
            hook()

    @torch.no_grad()
    def validation_step(
        self,
        batch,
    ) -> float:
        loss, _ = self._forward(batch)

        if (
            self.config.stop_on_nonfinite_loss
            and not torch.isfinite(loss)
        ):
            raise FloatingPointError(
                f"Non-finite validation loss: "
                f"{loss.detach().item()}"
            )

        return float(
            loss.detach().float().item()
        )

    def train_epoch(self) -> float:
        self.model.train()

        losses = []

        iterator = self._iterate_loader(
            self.train_loader,
            self.config.train_iterations_per_epoch,
        )

        total = self.config.train_iterations_per_epoch

        if total is not None:
            iterator = tqdm(
                iterator,
                total=total,
                desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}",
                disable=not self.is_main_process,
                dynamic_ncols=True,
            )

        for batch in iterator:
            loss = self.train_step(batch)
            losses.append(loss)

            if self.is_main_process:
                iterator.set_postfix(
                    loss=f"{loss:.4f}"
                )

        if not losses:
            raise RuntimeError(
                "Training epoch produced no batches"
            )

        mean_loss = sum(losses) / len(losses)

        return self._reduce_mean(
            mean_loss
        )

    def validate_epoch(
        self,
    ) -> float | None:
        if self.val_loader is None:
            return None

        self.model.eval()

        losses = []

        iterator = self._iterate_loader(
            self.val_loader,
            self.config.val_iterations_per_epoch,
        )

        for batch in iterator:
            loss = self.validation_step(batch)
            losses.append(loss)

        if not losses:
            raise RuntimeError(
                "Validation epoch produced no batches"
            )

        mean_loss = sum(losses) / len(losses)

        return self._reduce_mean(
            mean_loss
        )

    def _current_lr(self) -> float:
        return float(
            self.optimizer.param_groups[0]["lr"]
        )

    def _scheduler_step(self):
        if self.scheduler is None:
            return

        # FM-Bench schedulers accept explicit epoch/step.
        try:
            self.scheduler.step(
                self.current_epoch
            )
        except TypeError:
            self.scheduler.step()

    def _method_extra_state(self) -> dict:
        model = self.model

        if isinstance(model, DDP):
            model = model.module

        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        if hasattr(
            model,
            "get_checkpoint_state",
        ):
            return model.get_checkpoint_state()

        return {}

    def _restore_method_extra_state(
        self,
        state: dict,
    ):
        model = self.model

        if isinstance(model, DDP):
            model = model.module

        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        if (
            state
            and hasattr(
                model,
                "load_checkpoint_state",
            )
        ):
            model.load_checkpoint_state(
                state
            )

    def save(
        self,
        filename: str,
    ):
        if not self.is_main_process:
            return None

        return save_checkpoint(
            self.output_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            history=self.history,
            config=asdict(self.config),
            extra_state=self._method_extra_state(),
        )

    def resume(
        self,
        checkpoint_path: str | Path,
        reset_optimizer_scheduler: bool = False,
    ):
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=(
                None
                if reset_optimizer_scheduler
                else self.optimizer
            ),
            scheduler=(
                None
                if reset_optimizer_scheduler
                else self.scheduler
            ),
            scaler=self.scaler,
            map_location=self.device,
        )

        self.current_epoch = int(
            checkpoint["epoch"]
        )

        self.global_step = int(
            checkpoint["global_step"]
        )

        self.best_val_loss = checkpoint.get(
            "best_val_loss"
        )

        self.history = checkpoint.get(
            "history",
            self.history,
        )

        self._restore_method_extra_state(
            checkpoint.get(
                "extra_state",
                {},
            )
        )

        return checkpoint

    def _write_log(
        self,
        train_loss: float,
        val_loss: float | None,
    ):
        """Append one completed epoch to the training log."""

        if not self.is_main_process:
            return

        path = self.output_dir / "training_log.txt"

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        val_text = (
            f"{val_loss:.6f}"
            if val_loss is not None
            else "NA"
        )

        line = (
            f"{timestamp} | "
            f"epoch={self.current_epoch:04d}/"
            f"{self.config.epochs:04d} | "
            f"global_step={self.global_step} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_text} | "
            f"lr={self._current_lr():.8e}\n"
        )

        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def _write_history(self):
        if not self.is_main_process:
            return

        path = (
            self.output_dir
            / "history.json"
        )

        with open(path, "w") as f:
            json.dump(
                self.history,
                f,
                indent=2,
            )

    def fit(self):
        """Run training from current_epoch to config.epochs."""

        try:
            while (
                self.current_epoch
                < self.config.epochs
            ):
                self._scheduler_step()

                train_loss = self.train_epoch()
                val_loss = self.validate_epoch()

                self.history[
                    "train_loss"
                ].append(train_loss)

                self.history[
                    "val_loss"
                ].append(val_loss)

                self.history[
                    "lr"
                ].append(
                    self._current_lr()
                )

                # Store the next epoch to run.
                self.current_epoch += 1

                self._write_log(
                    train_loss,
                    val_loss,
                )

                if (
                    self.current_epoch
                    % self.config.checkpoint_every
                    == 0
                ):
                    self.save(
                        "checkpoint_latest.pth"
                    )

                if val_loss is not None:
                    if (
                        self.best_val_loss is None
                        or val_loss
                        < self.best_val_loss
                    ):
                        self.best_val_loss = (
                            val_loss
                        )

                        self.save(
                            "checkpoint_best.pth"
                        )

                self._write_history()

                if self.exit_training:
                    self.save(
                        "checkpoint_latest.pth"
                    )
                    break

            if (
                self.current_epoch
                >= self.config.epochs
            ):
                self.save(
                    "checkpoint_final.pth"
                )

            return self.history

        except KeyboardInterrupt:
            self.save(
                "checkpoint_latest.pth"
            )
            raise
