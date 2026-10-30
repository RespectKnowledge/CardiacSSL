"""Checkpoint utilities for FM-Bench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model from common wrappers."""

    if isinstance(model, DistributedDataParallel):
        model = model.module

    # torch.compile wrapper.
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    return model


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    best_val_loss: float | None,
    history: dict | None = None,
    config: dict | None = None,
    extra_state: dict | None = None,
) -> dict:
    """Construct a complete FM-Bench training checkpoint."""

    model = unwrap_model(model)

    return {
        "model_state": model.state_dict(),
        "optimizer_state": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "scheduler_state": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler_state": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": best_val_loss,
        "history": history or {},
        "config": config or {},
        "extra_state": extra_state or {},
    }


def save_checkpoint(
    path: str | Path,
    **checkpoint_kwargs,
) -> Path:
    """Save an FM-Bench checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = build_checkpoint(
        **checkpoint_kwargs
    )

    # Write to a temporary file first so an interrupted
    # checkpoint write cannot corrupt the last valid checkpoint.
    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """Restore model/training state from an FM-Bench checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    model_to_load = unwrap_model(model)

    state_dict = checkpoint["model_state"]

    # Compatibility with checkpoints saved from DDP.
    cleaned_state = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]

        cleaned_state[key] = value

    model_to_load.load_state_dict(
        cleaned_state,
        strict=strict,
    )

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state") is not None
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state") is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state"]
        )

    if (
        scaler is not None
        and checkpoint.get("scaler_state") is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state"]
        )

    return checkpoint
