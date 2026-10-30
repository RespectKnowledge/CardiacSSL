"""3D SimCLR view generation for FM-Bench."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cardiac_ssl.transforms.common import MRIAugmentation


def _tuple3(values: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(v) for v in values)

    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(
            f"Expected positive 3D size, got {values}"
        )

    return values


class SimCLRTransform3D:
    """Generate paired partially-overlapping 3D MRI views.

    The implementation follows the existing nnSSL SimCLR data semantics:
    reference crops are sampled first, then target crops are sampled from
    nearby spatial locations, after which both crop sets are independently
    augmented.
    """

    def __init__(
        self,
        crop_size: Sequence[int],
        crop_count_per_image: int = 1,
        min_overlap_ratio: float = 0.5,
        augment: bool = True,
        seed: int | None = None,
    ):
        self.crop_size = _tuple3(crop_size)

        if crop_count_per_image <= 0:
            raise ValueError("crop_count_per_image must be positive")

        if not 0.0 <= min_overlap_ratio <= 1.0:
            raise ValueError(
                "min_overlap_ratio must be between 0 and 1"
            )

        self.crop_count_per_image = int(crop_count_per_image)
        self.min_overlap_ratio = float(min_overlap_ratio)

        # Retain the offset semantics of the existing nnSSL implementation.
        self.max_offset_per_axis = tuple(
            int(size * self.min_overlap_ratio)
            for size in self.crop_size
        )

        self.augmentation = MRIAugmentation(enabled=augment)
        self.rng = np.random.default_rng(seed)

    def _validate_input(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Expected NumPy array, got {type(data).__name__}"
            )

        if data.ndim != 5:
            raise ValueError(
                f"Expected (B,C,D,H,W), got {data.shape}"
            )

        spatial_shape = data.shape[2:]

        for image_dim, crop_dim in zip(
            spatial_shape,
            self.crop_size,
        ):
            if image_dim < crop_dim:
                raise ValueError(
                    "SimCLR input must be at least as large as "
                    f"crop_size. Input={spatial_shape}, "
                    f"crop={self.crop_size}"
                )

    def _random_reference_offset(
        self,
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int]:
        return tuple(
            int(self.rng.integers(0, image_dim - crop_dim + 1))
            for image_dim, crop_dim in zip(
                spatial_shape,
                self.crop_size,
            )
        )

    def _overlapping_offset(
        self,
        reference: Sequence[int],
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int]:
        result = []

        for ref, image_dim, crop_dim, max_offset in zip(
            reference,
            spatial_shape,
            self.crop_size,
            self.max_offset_per_axis,
        ):
            low = max(0, int(ref) - max_offset)
            high = min(
                image_dim - crop_dim,
                int(ref) + max_offset,
            )

            result.append(
                int(self.rng.integers(low, high + 1))
            )

        return tuple(result)

    def _crop(
        self,
        volume: np.ndarray,
        offset: Sequence[int],
    ) -> np.ndarray:
        d, h, w = (int(v) for v in offset)
        pd, ph, pw = self.crop_size

        return volume[
            :,
            d:d + pd,
            h:h + ph,
            w:w + pw,
        ]

    def reseed_from_worker(self) -> None:
        """Create a private RNG from the worker-specific NumPy state."""
        worker_seed = int(
            np.random.randint(0, np.iinfo(np.uint32).max)
        )
        self.rng = np.random.default_rng(worker_seed)

    def __call__(self, data: np.ndarray) -> dict:
        self._validate_input(data)

        batch_size = data.shape[0]
        spatial_shape = data.shape[2:]

        reference_crops = []
        target_crops = []

        reference_offsets = []
        target_offsets = []

        for batch_index in range(batch_size):
            volume = data[batch_index]

            for _ in range(self.crop_count_per_image):
                ref_offset = self._random_reference_offset(
                    spatial_shape
                )

                target_offset = self._overlapping_offset(
                    ref_offset,
                    spatial_shape,
                )

                reference_crops.append(
                    self._crop(volume, ref_offset)
                )

                target_crops.append(
                    self._crop(volume, target_offset)
                )

                reference_offsets.append(ref_offset)
                target_offsets.append(target_offset)

        reference_crops = np.stack(
            reference_crops,
            axis=0,
        ).astype(np.float32, copy=False)

        target_crops = np.stack(
            target_crops,
            axis=0,
        ).astype(np.float32, copy=False)

        # Independent augmentation of the two views.
        reference_crops = self.augmentation(reference_crops)
        target_crops = self.augmentation(target_crops)

        all_crops = np.concatenate(
            [reference_crops, target_crops],
            axis=0,
        )

        return {
            "all_crops": np.ascontiguousarray(all_crops),
            "reference_crops": reference_crops,
            "target_crops": target_crops,
            "reference_offsets": np.asarray(
                reference_offsets,
                dtype=np.int64,
            ),
            "target_offsets": np.asarray(
                target_offsets,
                dtype=np.int64,
            ),
            "reference_crop_index": len(reference_crops),
            "batch_size": batch_size,
            "n_crops_per_image": self.crop_count_per_image,
        }
