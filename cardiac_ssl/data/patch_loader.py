"""Efficient patch-wise loading for preprocessed 3D volumes.

This follows the nnSSL/nnU-Net loading strategy:
open the Blosc2 array lazily, determine a random bounding box, read only
the valid part of that box, then pad only when the box extends outside
the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from cardiac_ssl.data.dataset import FMBenchDataset


def sample_bbox(
    image_shape: Sequence[int],
    patch_size: Sequence[int],
    final_patch_size: Sequence[int],
    rng: np.random.Generator | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Sample an nnSSL-style random patch bounding box."""

    if rng is None:
        rng = np.random.default_rng()

    image_shape = np.asarray(image_shape, dtype=int)
    patch_size = np.asarray(patch_size, dtype=int)
    final_patch_size = np.asarray(final_patch_size, dtype=int)

    if not (
        len(image_shape) == len(patch_size) == len(final_patch_size) == 3
    ):
        raise ValueError("Expected three spatial dimensions.")

    if np.any(patch_size < final_patch_size):
        raise ValueError(
            "patch_size must be >= final_patch_size in every dimension."
        )

    need_to_pad = patch_size - final_patch_size

    for d in range(3):
        if need_to_pad[d] + image_shape[d] < patch_size[d]:
            need_to_pad[d] = patch_size[d] - image_shape[d]

    lbs = -(need_to_pad // 2)

    ubs = (
        image_shape
        + need_to_pad // 2
        + need_to_pad % 2
        - patch_size
    )

    bbox_lbs = np.asarray(
        [
            rng.integers(int(lbs[d]), int(ubs[d]) + 1)
            for d in range(3)
        ],
        dtype=int,
    )

    bbox_ubs = bbox_lbs + patch_size

    return tuple(bbox_lbs.tolist()), tuple(bbox_ubs.tolist())


def load_patch(
    image_path: str | Path,
    patch_size: Sequence[int],
    final_patch_size: Sequence[int],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Read only the requested spatial patch from a Blosc2 volume."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    arr = FMBenchDataset._open_blosc2(image_path)

    if len(arr.shape) == 3:
        raise ValueError(
            "FM-Bench preprocessed volumes must have shape (C,D,H,W)."
        )

    if len(arr.shape) != 4:
        raise ValueError(
            f"Expected (C,D,H,W), got shape {tuple(arr.shape)}"
        )

    image_shape = tuple(int(x) for x in arr.shape[1:])

    bbox_lbs, bbox_ubs = sample_bbox(
        image_shape=image_shape,
        patch_size=patch_size,
        final_patch_size=final_patch_size,
        rng=rng,
    )

    valid_lbs = tuple(
        max(0, bbox_lbs[d]) for d in range(3)
    )
    valid_ubs = tuple(
        min(image_shape[d], bbox_ubs[d]) for d in range(3)
    )

    slices = (
        slice(None),
        slice(valid_lbs[0], valid_ubs[0]),
        slice(valid_lbs[1], valid_ubs[1]),
        slice(valid_lbs[2], valid_ubs[2]),
    )

    # Critical performance point:
    # only this region is decompressed/materialized.
    data = np.asarray(arr[slices], dtype=np.float32)

    padding = tuple(
        (
            -min(0, bbox_lbs[d]),
            max(bbox_ubs[d] - image_shape[d], 0),
        )
        for d in range(3)
    )

    if any(left or right for left, right in padding):
        data = np.pad(
            data,
            ((0, 0), *padding),
            mode="constant",
            constant_values=0,
        )

    expected = tuple(int(x) for x in patch_size)

    if tuple(data.shape[1:]) != expected:
        raise RuntimeError(
            f"Patch loader produced {data.shape[1:]}, "
            f"expected {expected}"
        )

    return data, (bbox_lbs, bbox_ubs)
