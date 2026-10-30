"""nnSSL-style spatial augmentation utilities for FM-Bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from batchgenerators.augmentations.utils import rotate_coords_3d
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)


def _tuple3(values: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(v) for v in values)

    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {values}")

    if any(v <= 0 for v in values):
        raise ValueError(f"Values must be positive, got {values}")

    return values


def compute_initial_patch_size(
    final_patch_size: Sequence[int],
    rotation_x: tuple[float, float],
    rotation_y: tuple[float, float],
    rotation_z: tuple[float, float],
    scale_range: tuple[float, float] = (0.85, 1.25),
) -> tuple[int, int, int]:
    """Compute enlarged patch size required before spatial augmentation.

    This reproduces the nnSSL/nnU-Net-style initial-patch calculation.
    """

    final_patch_size = np.asarray(
        _tuple3(final_patch_size),
        dtype=float,
    )

    def max_rotation(rotation):
        return min(
            np.pi / 2,
            float(np.max(np.abs(rotation))),
        )

    rx = max_rotation(rotation_x)
    ry = max_rotation(rotation_y)
    rz = max_rotation(rotation_z)

    final_shape = final_patch_size.copy()

    final_shape = np.maximum(
        np.abs(
            rotate_coords_3d(
                final_patch_size,
                rx,
                0,
                0,
            )
        ),
        final_shape,
    )

    final_shape = np.maximum(
        np.abs(
            rotate_coords_3d(
                final_patch_size,
                0,
                ry,
                0,
            )
        ),
        final_shape,
    )

    final_shape = np.maximum(
        np.abs(
            rotate_coords_3d(
                final_patch_size,
                0,
                0,
                rz,
            )
        ),
        final_shape,
    )

    final_shape /= min(scale_range)

    return tuple(final_shape.astype(int))


@dataclass(frozen=True)
class SpatialAugmentationConfig:
    """Spatial augmentation configuration."""

    final_patch_size: tuple[int, int, int] = (128, 128, 128)

    rotation_x: tuple[float, float] = (
        -np.pi / 6,
        np.pi / 6,
    )
    rotation_y: tuple[float, float] = (
        -np.pi / 6,
        np.pi / 6,
    )
    rotation_z: tuple[float, float] = (
        -np.pi / 6,
        np.pi / 6,
    )

    scale_range: tuple[float, float] = (0.85, 1.25)

    mirror_axes: tuple[int, int, int] = (0, 1, 2)

    p_rotation: float = 0.2
    p_scaling: float = 0.2

    @property
    def initial_patch_size(self) -> tuple[int, int, int]:
        return compute_initial_patch_size(
            self.final_patch_size,
            self.rotation_x,
            self.rotation_y,
            self.rotation_z,
            self.scale_range,
        )


def build_spatial_augmentation(
    config: SpatialAugmentationConfig,
) -> Compose:
    """Build 3D spatial augmentation producing the final patch size."""

    return Compose(
        [
            SpatialTransform(
                patch_size=config.final_patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0.0, 0.0),
                sigma=(0.0, 0.0),
                do_rotation=True,
                angle_x=config.rotation_x,
                angle_y=config.rotation_y,
                angle_z=config.rotation_z,
                p_rot_per_axis=1.0,
                do_scale=True,
                scale=config.scale_range,
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=False,
                p_el_per_sample=0.0,
                p_scale_per_sample=config.p_scaling,
                p_rot_per_sample=config.p_rotation,
            ),
            MirrorTransform(
                axes=config.mirror_axes,
            ),
        ]
    )


class SpatialAugmentation3D:
    """Apply nnSSL-style spatial augmentation to a 3D MRI batch.

    Input:
        (B, C, D, H, W), normally sampled at initial_patch_size.

    Output:
        (B, C, *final_patch_size)
    """

    def __init__(
        self,
        config: SpatialAugmentationConfig | None = None,
        enabled: bool = True,
    ):
        self.config = config or SpatialAugmentationConfig()
        self.enabled = enabled

        self.transform = (
            build_spatial_augmentation(self.config)
            if enabled
            else None
        )

    @property
    def initial_patch_size(self) -> tuple[int, int, int]:
        return self.config.initial_patch_size

    @property
    def final_patch_size(self) -> tuple[int, int, int]:
        return self.config.final_patch_size

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Expected NumPy array, got {type(data).__name__}"
            )

        if data.ndim != 5:
            raise ValueError(
                f"Expected (B,C,D,H,W), got {data.shape}"
            )

        if not np.isfinite(data).all():
            raise ValueError("Input contains NaN or Inf")

        if not self.enabled:
            return data.copy()

        output = self.transform(
            **{
                "data": data.copy(),
                "seg": None,
            }
        )["data"]

        return np.ascontiguousarray(output)
