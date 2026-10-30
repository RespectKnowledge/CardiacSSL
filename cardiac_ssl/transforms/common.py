"""Common 3D MRI augmentations used by FM-Bench SSL methods."""

from __future__ import annotations

import numpy as np

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    Rot90Transform,
)


def build_mri_intensity_augmentation() -> Compose:
    """Build the nnSSL-style augmentation used for MRI SSL crops.

    This preserves the augmentation probabilities/ranges used by the
    existing nnSSL SimCLR pipeline.
    """

    return Compose(
        [
            GaussianNoiseTransform(
                p_per_sample=0.1,
            ),
            GaussianBlurTransform(
                (0.5, 1.0),
                different_sigma_per_channel=True,
                p_per_sample=0.2,
                p_per_channel=0.5,
            ),
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25),
                p_per_sample=0.15,
            ),
            ContrastAugmentationTransform(
                p_per_sample=0.15,
            ),
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1.0),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.1,
                ignore_axes=None,
            ),
            GammaTransform(
                (0.7, 1.5),
                True,
                True,
                retain_stats=True,
                p_per_sample=0.1,
            ),
            GammaTransform(
                (0.7, 1.5),
                False,
                True,
                retain_stats=True,
                p_per_sample=0.3,
            ),
            MirrorTransform(axes=(0,)),
            MirrorTransform(axes=(1,)),
            MirrorTransform(axes=(2,)),
            # Cardiac SAX crops are anisotropic (D << H,W).
            # Rotate only within the in-plane H-W plane so that
            # spatial dimensions are preserved.
            Rot90Transform(axes=(1, 2)),
        ]
    )


class MRIAugmentation:
    """Apply common MRI augmentation to a NumPy batch.

    Input shape:
        (B, C, D, H, W)

    Output shape:
        (B, C, D, H, W)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.transform = (
            build_mri_intensity_augmentation()
            if enabled
            else None
        )

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

        result = self.transform(
            **{
                "data": data.copy(),
                "seg": None,
            }
        )["data"]

        return np.ascontiguousarray(result)
