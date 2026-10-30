from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DINOv2Config:
    out_dim: int = 4096
    hidden_dim: int = 2048
    bottleneck_dim: int = 256

    student_temperature: float = 0.1
    teacher_temperature: float = 0.04
    teacher_momentum: float = 0.996

    center_momentum: float = 0.9


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class DINOv2(nn.Module):
    """
    DINO-style self-distillation using the existing Swin3D backbone.

    Student:
        Swin3D -> pooled feature -> DINO head

    Teacher:
        EMA copy of student backbone/head

    Teacher receives global crops.
    Student receives global + local crops.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: DINOv2Config | None = None,
    ):
        super().__init__()

        self.config = config or DINOv2Config()

        if not hasattr(backbone, "feature_dim"):
            raise TypeError(
                "DINO backbone must expose feature_dim."
            )

        if not callable(
            getattr(backbone, "forward_pooled", None)
        ):
            raise TypeError(
                "DINO backbone must implement forward_pooled()."
            )

        self.backbone = backbone

        self.student_head = DINOHead(
            in_dim=int(backbone.feature_dim),
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            out_dim=self.config.out_dim,
        )

        self.teacher_backbone = deepcopy(backbone)

        self.teacher_head = DINOHead(
            in_dim=int(backbone.feature_dim),
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            out_dim=self.config.out_dim,
        )
        self.teacher_head.load_state_dict(
            self.student_head.state_dict()
        )

        for p in self.teacher_backbone.parameters():
            p.requires_grad_(False)

        for p in self.teacher_head.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "center",
            torch.zeros(1, self.config.out_dim),
        )

    def train(self, mode=True):
        super().train(mode)

        # Teacher must always remain deterministic.
        self.teacher_backbone.eval()
        self.teacher_head.eval()

        return self

    def _student(self, x):
        features = self.backbone.forward_pooled(x)
        return self.student_head(features)

    @torch.no_grad()
    def _teacher(self, x):
        features = self.teacher_backbone.forward_pooled(x)
        return self.teacher_head(features)

    def _teacher_probabilities(self, logits):
        return F.softmax(
            (logits - self.center)
            / self.config.teacher_temperature,
            dim=-1,
        )

    def _student_log_probabilities(self, logits):
        return F.log_softmax(
            logits / self.config.student_temperature,
            dim=-1,
        )

    def forward(self, batch):
        if not isinstance(batch, dict):
            raise TypeError(
                "DINOv2 expects a batch dictionary."
            )

        global_crops = batch.get("global_crops")
        local_crops = batch.get("local_crops", [])

        if global_crops is None:
            raise KeyError(
                "DINOv2 batch requires global_crops."
            )

        if len(global_crops) != 2:
            raise ValueError(
                "DINOv2 currently expects exactly two global crops."
            )

        # -----------------------------------
        # Teacher: global views only
        # -----------------------------------

        with torch.no_grad():
            teacher_logits = [
                self._teacher(x)
                for x in global_crops
            ]

            teacher_probs = [
                self._teacher_probabilities(x)
                for x in teacher_logits
            ]

        # -----------------------------------
        # Student: global + local views
        # -----------------------------------

        student_views = (
            list(global_crops)
            + list(local_crops)
        )

        student_logits = [
            self._student(x)
            for x in student_views
        ]

        student_log_probs = [
            self._student_log_probabilities(x)
            for x in student_logits
        ]

        # -----------------------------------
        # Cross-view DINO loss
        #
        # Do not match identical global view
        # student/teacher pairs.
        # -----------------------------------

        total_loss = 0.0
        n_terms = 0

        for teacher_index, teacher_prob in enumerate(
            teacher_probs
        ):
            for student_index, student_log_prob in enumerate(
                student_log_probs
            ):
                if student_index == teacher_index:
                    continue

                loss = -(
                    teacher_prob
                    * student_log_prob
                ).sum(dim=-1).mean()

                total_loss = total_loss + loss
                n_terms += 1

        if n_terms == 0:
            raise RuntimeError(
                "No valid DINO cross-view loss terms."
            )

        loss = total_loss / n_terms

        # -----------------------------------
        # Update teacher output center
        # -----------------------------------

        with torch.no_grad():
            batch_center = torch.cat(
                teacher_logits,
                dim=0,
            ).mean(dim=0, keepdim=True)

            self.center.mul_(
                self.config.center_momentum
            ).add_(
                batch_center,
                alpha=1.0
                - self.config.center_momentum,
            )

        return {
            "loss": loss,
            "loss_dino": loss.detach(),
        }

    @torch.no_grad()
    def on_after_optimizer_step(self):
        """
        EMA student -> teacher update.
        Called automatically by existing Trainer.
        """

        momentum = self.config.teacher_momentum

        for student, teacher in zip(
            self.backbone.parameters(),
            self.teacher_backbone.parameters(),
        ):
            teacher.data.mul_(momentum).add_(
                student.data,
                alpha=1.0 - momentum,
            )

        for student, teacher in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters(),
        ):
            teacher.data.mul_(momentum).add_(
                student.data,
                alpha=1.0 - momentum,
            )

    def get_checkpoint_state(self):
        return {
            "center": self.center.detach().cpu(),
        }

    def load_checkpoint_state(self, state):
        if "center" in state:
            self.center.copy_(
                state["center"].to(self.center.device)
            )
