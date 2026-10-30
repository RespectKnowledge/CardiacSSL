"""
3D Swin Transformer encoder for cardiac MRI self-supervised learning.

The implementation uses MONAI's SwinTransformer and exposes hierarchical
feature maps for multi-level teacher-student alignment and downstream
segmentation.

Input:
    [B, C, D, H, W]

Outputs:
    Five hierarchical feature maps:
        x0, x1, x2, x3, x4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from monai.networks.nets.swin_unetr import SwinTransformer


def _triple(x: Sequence[int]) -> tuple[int, int, int]:
    if len(x) != 3:
        raise ValueError(
            f"Expected three spatial dimensions, got {tuple(x)}."
        )
    return tuple(int(v) for v in x)


@dataclass(frozen=True)
class Swin3DConfig:
    """
    Configuration of the cardiac 3D Swin Transformer encoder.

    The architecture is intentionally encoder-only so that exactly the
    same encoder can be used for SwinDINO-3D, MAE, SimCLR and V-JEPA.
    """

    input_channels: int = 1

    # Swin patch embedding.
    # This is NOT the 64^3 training crop.
    patch_size: tuple[int, int, int] = (2, 2, 2)

    # Local self-attention window.
    window_size: tuple[int, int, int] = (7, 7, 7)

    # Hierarchical Swin configuration.
    embed_dim: int = 48
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)

    mlp_ratio: float = 4.0
    qkv_bias: bool = True

    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    patch_norm: bool = False
    use_checkpoint: bool = False

    downsample: str = "merging"
    use_v2: bool = False

    normalize: bool = True


class Swin3D(nn.Module):
    """
    Hierarchical MONAI 3D Swin Transformer encoder.

    The network exposes:

        forward_intermediates(x)
            -> [x0, x1, x2, x3, x4]

        forward_features(x)
            -> x4

        forward_pooled(x)
            -> global representation

    The multi-scale outputs are required by the proposed cardiac
    teacher-student SSL framework.
    """

    def __init__(
        self,
        config: Swin3DConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or Swin3DConfig()

        self.patch_size = _triple(self.config.patch_size)
        self.window_size = _triple(self.config.window_size)

        if self.config.input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        if self.config.embed_dim <= 0:
            raise ValueError(
                "embed_dim must be positive."
            )

        if len(self.config.depths) != 4:
            raise ValueError(
                "Cardiac Swin3D expects four hierarchical stages."
            )

        if len(self.config.num_heads) != 4:
            raise ValueError(
                "Cardiac Swin3D expects four num_heads values."
            )

        self.encoder = SwinTransformer(
            in_chans=self.config.input_channels,
            embed_dim=self.config.embed_dim,
            window_size=self.window_size,
            patch_size=self.patch_size,
            depths=self.config.depths,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            qkv_bias=self.config.qkv_bias,
            drop_rate=self.config.drop_rate,
            attn_drop_rate=self.config.attn_drop_rate,
            drop_path_rate=self.config.drop_path_rate,
            patch_norm=self.config.patch_norm,
            use_checkpoint=self.config.use_checkpoint,
            spatial_dims=3,
            downsample=self.config.downsample,
            use_v2=self.config.use_v2,
        )

        # MONAI hierarchical feature dimensions:
        #
        # x0 = E
        # x1 = 2E
        # x2 = 4E
        # x3 = 8E
        # x4 = 16E
        #
        # With E=48:
        # [48, 96, 192, 384, 768]

        self.output_channels = [
            self.config.embed_dim * (2 ** i)
            for i in range(5)
        ]

        self.feature_dim = self.output_channels[-1]

        # patch embedding gives /2 and the four hierarchical
        # downsampling operations give the final /32 resolution.
        self.downsample_factor = tuple(
            p * (2 ** len(self.config.depths))
            for p in self.patch_size
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:

        if x.ndim != 5:
            raise ValueError(
                "Expected cardiac MRI tensor [B,C,D,H,W], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} channel(s), "
                f"got {x.shape[1]}."
            )

        spatial = tuple(int(v) for v in x.shape[2:])

        if any(
            size % factor != 0
            for size, factor in zip(
                spatial,
                self.downsample_factor,
            )
        ):
            raise ValueError(
                f"Input {spatial} must be divisible by "
                f"{self.downsample_factor}."
            )

    def forward_intermediates(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Return hierarchical Swin feature maps.

        For a 64^3 input with embed_dim=48:

            x0 : [B,  48, 32, 32, 32]
            x1 : [B,  96, 16, 16, 16]
            x2 : [B, 192,  8,  8,  8]
            x3 : [B, 384,  4,  4,  4]
            x4 : [B, 768,  2,  2,  2]
        """

        self._validate_input(x)

        return self.encoder(
            x,
            normalize=self.config.normalize,
        )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return final hierarchical feature map."""

        return self.forward_intermediates(x)[-1]

    def forward_pooled(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return global representation used by SSL methods
        such as SimCLR.
        """

        x = self.forward_features(x)
        x = self.global_pool(x)

        return torch.flatten(x, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features(x)
