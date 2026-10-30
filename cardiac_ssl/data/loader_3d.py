"""3D patch sampling utilities for FM-Bench.

Implements nnSSL-style random spatial crop extraction with zero-padding for
volumes smaller than, or crops extending beyond, the requested patch size.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_3tuple(values: Sequence[int], name: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in values)

    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {values}")

    if any(v <= 0 for v in values):
        raise ValueError(f"{name} values must be positive, got {values}")

    return values


class PatchSampler3D:
    """Randomly sample fixed-size 3D patches from channel-first volumes."""

    def __init__(
        self,
        patch_size: Sequence[int],
        seed: int | None = None,
    ):
        self.patch_size = _as_3tuple(patch_size, "patch_size")
        self.rng = np.random.default_rng(seed)

    def sample_bbox(
        self,
        image_shape: Sequence[int],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Sample the lower and upper coordinates of a patch.

        Coordinates may extend outside the image when an image dimension is
        smaller than the requested patch. `extract_patch` handles this by
        cropping the valid region and zero-padding the missing area.
        """

        image_shape = _as_3tuple(image_shape, "image_shape")

        lower = []
        upper = []

        for image_dim, patch_dim in zip(image_shape, self.patch_size):
            if image_dim >= patch_dim:
                lb = int(self.rng.integers(0, image_dim - patch_dim + 1))
            else:
                # Allow the smaller image to appear at a random position
                # inside the output patch rather than always centering it.
                missing = patch_dim - image_dim
                lb = -int(self.rng.integers(0, missing + 1))

            lower.append(lb)
            upper.append(lb + patch_dim)

        return tuple(lower), tuple(upper)

    def extract_patch(
        self,
        data: np.ndarray,
        bbox: tuple[
            tuple[int, int, int],
            tuple[int, int, int],
        ],
    ) -> np.ndarray:
        """Extract a patch and zero-pad regions outside the source image.

        Args:
            data:
                Channel-first MRI array with shape (C, D, H, W).
            bbox:
                Tuple ``(lower, upper)`` containing 3D spatial coordinates.

        Returns:
            Array with shape ``(C, *patch_size)``.
        """

        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"data must be a NumPy array, got {type(data).__name__}"
            )

        if data.ndim != 4:
            raise ValueError(
                f"Expected data shape (C,D,H,W), got {data.shape}"
            )

        lower, upper = bbox

        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("bbox must contain 3D lower and upper coordinates")

        spatial_shape = data.shape[1:]

        src_slices = []
        dst_slices = []

        for lb, ub, image_dim, patch_dim in zip(
            lower,
            upper,
            spatial_shape,
            self.patch_size,
        ):
            src_start = max(0, lb)
            src_end = min(image_dim, ub)

            dst_start = src_start - lb
            dst_end = dst_start + (src_end - src_start)

            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))

        patch = np.zeros(
            (data.shape[0], *self.patch_size),
            dtype=data.dtype,
        )

        patch[
            :,
            dst_slices[0],
            dst_slices[1],
            dst_slices[2],
        ] = data[
            :,
            src_slices[0],
            src_slices[1],
            src_slices[2],
        ]

        return patch

    def __call__(
        self,
        data: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[tuple[int, int, int], tuple[int, int, int]],
    ]:
        """Sample and extract one random patch."""

        if data.ndim != 4:
            raise ValueError(
                f"Expected data shape (C,D,H,W), got {data.shape}"
            )

        bbox = self.sample_bbox(data.shape[1:])
        patch = self.extract_patch(data, bbox)

        return patch, bbox
