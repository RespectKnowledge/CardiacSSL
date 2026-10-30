"""Common backbone interface for FM-Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import nn


class Backbone3D(nn.Module, ABC):
    """Base interface shared by all FM-Bench 3D backbones."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Number of channels/features in the final representation."""
        raise NotImplementedError

    @property
    @abstractmethod
    def downsample_factor(self) -> tuple[int, int, int]:
        """Total spatial downsampling factor."""
        raise NotImplementedError

    @abstractmethod
    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final spatial/token feature representation."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        """Convert backbone features into one vector per sample.

        CNN feature maps:
            (B, C, D, H, W) -> (B, C)

        Transformer tokens:
            (B, N, C) -> (B, C)
        """

        if features.ndim == 5:
            return features.mean(dim=(2, 3, 4))

        if features.ndim == 3:
            return features.mean(dim=1)

        if features.ndim == 2:
            return features

        raise ValueError(
            "Expected features with shape (B,C,D,H,W), "
            "(B,N,C), or (B,C); "
            f"received {tuple(features.shape)}"
        )

    def forward_pooled(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.pool(
            self.forward_features(x)
        )


def validate_3d_input(
    x: torch.Tensor,
    channels: int,
) -> None:
    if not torch.is_tensor(x):
        raise TypeError("Input must be a torch.Tensor")

    if x.ndim != 5:
        raise ValueError(
            "Expected input shape (B,C,D,H,W), "
            f"received {tuple(x.shape)}"
        )

    if x.shape[1] != channels:
        raise ValueError(
            f"Expected {channels} input channels, "
            f"received {x.shape[1]}"
        )

    if min(x.shape[2:]) <= 0:
        raise ValueError(
            "Spatial dimensions must be positive"
        )
