"""Learning-rate schedulers for FM-Bench."""

from __future__ import annotations

import math

from torch.optim import Optimizer


class PolyLRScheduler:
    """Polynomial learning-rate decay.

    Reproduces the nnSSL PolyLR schedule:

        lr = initial_lr * (1 - step / max_steps) ** exponent
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
    ):
        if initial_lr < 0:
            raise ValueError("initial_lr must be non-negative")

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        self.last_step = -1

    def step(self, current_step: int) -> float:
        current_step = int(current_step)

        if current_step < 0:
            raise ValueError("current_step must be non-negative")

        # Protect against numerical problems if called after training ends.
        bounded_step = min(current_step, self.max_steps)

        lr = self.initial_lr * (
            1.0 - bounded_step / self.max_steps
        ) ** self.exponent

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.last_step = current_step
        return lr

    def state_dict(self) -> dict:
        return {
            "initial_lr": self.initial_lr,
            "max_steps": self.max_steps,
            "exponent": self.exponent,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.initial_lr = float(state_dict["initial_lr"])
        self.max_steps = int(state_dict["max_steps"])
        self.exponent = float(state_dict["exponent"])
        self.last_step = int(state_dict["last_step"])

        if self.last_step >= 0:
            self.step(self.last_step)


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        start_lr: float,
        peak_lr: float,
        final_lr: float = 0.0,
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        if warmup_steps >= total_steps:
            raise ValueError(
                "warmup_steps must be smaller than total_steps"
            )

        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

        self.start_lr = float(start_lr)
        self.peak_lr = float(peak_lr)
        self.final_lr = float(final_lr)

        self.last_step = -1

    def get_lr(self, step: int) -> float:
        step = min(max(int(step), 0), self.total_steps)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = step / self.warmup_steps

            return self.start_lr + progress * (
                self.peak_lr - self.start_lr
            )

        decay_steps = self.total_steps - self.warmup_steps

        progress = (
            step - self.warmup_steps
        ) / decay_steps

        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return self.final_lr + (
            self.peak_lr - self.final_lr
        ) * cosine

    def step(self, current_step: int) -> float:
        lr = self.get_lr(current_step)

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.last_step = int(current_step)
        return lr

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "start_lr": self.start_lr,
            "peak_lr": self.peak_lr,
            "final_lr": self.final_lr,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.total_steps = int(state_dict["total_steps"])
        self.start_lr = float(state_dict["start_lr"])
        self.peak_lr = float(state_dict["peak_lr"])
        self.final_lr = float(state_dict["final_lr"])
        self.last_step = int(state_dict["last_step"])

        if self.last_step >= 0:
            self.step(self.last_step)
