"""Training dataset and batch collation for FM-Bench SSL methods."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from cardiac_ssl.data.loader_3d import PatchSampler3D
from cardiac_ssl.transforms.spatial import (
    SpatialAugmentation3D,
    SpatialAugmentationConfig,
    compute_initial_patch_size,
)
from cardiac_ssl.transforms.common import MRIAugmentation
from cardiac_ssl.transforms.simclr import SimCLRTransform3D
try:
    from cardiac_ssl.transforms.dinov2 import DINOv2Transform3D
except ModuleNotFoundError:
    DINOv2Transform3D = None


class SSLTrainingDataset(Dataset):
    """Wrap a case dataset with patch sampling and spatial augmentation.

    Input cases follow the FM-Bench convention:
        data: (C, D, H, W)

    Output samples also remain channel-first:
        data: (C, D, H, W)
    """

    def __init__(
        self,
        dataset: Dataset,
        initial_patch_size: Sequence[int] | None = None,
        final_patch_size: Sequence[int] = (128, 128, 128),
        spatial_augmentation: bool = True,
        intensity_augmentation: bool = True,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset

        self.final_patch_size = tuple(
            int(v) for v in final_patch_size
        )

        spatial_config = SpatialAugmentationConfig(
            final_patch_size=self.final_patch_size,
            p_rotation=0.2 if spatial_augmentation else 0.0,
            p_scaling=0.2 if spatial_augmentation else 0.0,
        )

        if initial_patch_size is None:
            if spatial_augmentation:
                self.initial_patch_size = compute_initial_patch_size(
                    self.final_patch_size,
                    spatial_config.rotation_x,
                    spatial_config.rotation_y,
                    spatial_config.rotation_z,
                    spatial_config.scale_range,
                )
            else:
                self.initial_patch_size = self.final_patch_size
        else:
            self.initial_patch_size = tuple(
                int(v) for v in initial_patch_size
            )

        self.patch_sampler = PatchSampler3D(
            self.initial_patch_size,
            seed=seed,
        )

        # Keep the spatial transform enabled even when random
        # rotation/scaling are disabled because it also performs
        # the required center crop to final_patch_size.
        self.spatial = SpatialAugmentation3D(
            spatial_config,
            enabled=True,
        )

        self.intensity = MRIAugmentation(
            enabled=intensity_augmentation,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        case = self.dataset[index]

        data = np.asarray(
            case["data"],
            dtype=np.float32,
        )

        if data.ndim != 4:
            raise ValueError(
                "SSLTrainingDataset expects case data "
                "with shape (C,D,H,W), got "
                f"{data.shape}."
            )

        patch, bbox = self.patch_sampler(data)

        # SpatialAugmentation3D works on a batch.
        patch = self.spatial(patch[None])[0]

        # Intensity transforms also operate on batched MRI data.
        patch = self.intensity(patch[None])[0]

        patch = np.ascontiguousarray(
            patch,
            dtype=np.float32,
        )

        return {
            "data": patch,
            "properties": case.get("properties"),
            "case_id": case.get("case_id"),
            "path": case.get("path"),
            "bbox": bbox,
        }


class SSLBatchCollator:
    """Create the method-specific batch expected by each SSL objective."""

    def __init__(
        self,
        method: str,
        crop_size: Sequence[int] = (128, 128, 128),
        simclr_crop_count_per_image: int = 1,
        simclr_min_overlap_ratio: float = 0.5,
        simclr_augment: bool = True,
        dinov2_local_crop_size: Sequence[int] | None = None,
        dinov2_n_local_crops: int = 4,
        dinov2_augment: bool = True,
        seed: int | None = None,
    ) -> None:
        self.method = method.lower().strip()

        if self.method not in {
            "mae",
            "simclr",
            "dinov2",
            "vjepa",
        }:
            raise ValueError(
                f"Unknown SSL method: {method!r}"
            )

        self.simclr = None
        self.dinov2 = None

        if self.method == "simclr":
            self.simclr = SimCLRTransform3D(
                crop_size=crop_size,
                crop_count_per_image=(
                    simclr_crop_count_per_image
                ),
                min_overlap_ratio=(
                    simclr_min_overlap_ratio
                ),
                augment=simclr_augment,
                seed=seed,
            )

        if self.method == "dinov2":
            if DINOv2Transform3D is None:
                raise ImportError(
                    "DINOv2Transform3D is unavailable. "
                    "Install/restore cardiac_ssl.transforms.dinov2 "
                    "before using method='dinov2'."
                )
            self.dinov2 = DINOv2Transform3D(
                global_crop_size=crop_size,
                local_crop_size=dinov2_local_crop_size,
                n_global_crops=2,
                n_local_crops=dinov2_n_local_crops,
                augment=dinov2_augment,
                seed=seed,
            )

    def reseed_from_worker(self) -> None:
        """Initialize method-specific RNGs from worker NumPy state."""
        if self.simclr is not None:
            self.simclr.reseed_from_worker()

        if self.dinov2 is not None:
            self.dinov2.reseed_from_worker()

    @staticmethod
    def _stack_data(
        samples: list[dict[str, Any]],
    ) -> np.ndarray:
        if not samples:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        return np.stack(
            [sample["data"] for sample in samples],
            axis=0,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _to_tensor(
        value: np.ndarray,
    ) -> torch.Tensor:
        return torch.from_numpy(
            np.ascontiguousarray(value)
        )

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        data = self._stack_data(samples)

        metadata = {
            "case_id": [
                sample.get("case_id")
                for sample in samples
            ],
            "path": [
                sample.get("path")
                for sample in samples
            ],
            "properties": [
                sample.get("properties")
                for sample in samples
            ],
        }

        if self.method == "dinov2":
            assert self.dinov2 is not None

            batch = self.dinov2(data)

            return {
                "global_crops": [
                    self._to_tensor(crop)
                    for crop in batch["global_crops"]
                ],
                "local_crops": [
                    self._to_tensor(crop)
                    for crop in batch["local_crops"]
                ],
                **metadata,
            }

        if self.method == "simclr":
            assert self.simclr is not None

            batch = self.simclr(data)

            return {
                "all_crops": self._to_tensor(
                    batch["all_crops"]
                ),
                "reference_crops": self._to_tensor(
                    batch["reference_crops"]
                ),
                "target_crops": self._to_tensor(
                    batch["target_crops"]
                ),
                "reference_offsets": self._to_tensor(
                    batch["reference_offsets"]
                ),
                "target_offsets": self._to_tensor(
                    batch["target_offsets"]
                ),
                "reference_crop_index": int(
                    batch["reference_crop_index"]
                ),
                "batch_size": int(
                    batch["batch_size"]
                ),
                "n_crops_per_image": int(
                    batch["n_crops_per_image"]
                ),
                **metadata,
            }

        # MAE, DINOv2 and V-JEPA currently consume the
        # standard image batch. Method-owned masking happens
        # inside the model.
        return {
            "data": self._to_tensor(data),
            **metadata,
        }



class SSLBatchTransform:
    """Worker-side transform for already-batched patch data."""

    def __init__(
        self,
        method: str,
        crop_size: Sequence[int],
        spatial_augmentation: bool = True,
        intensity_augmentation: bool = True,
        simclr_crop_count_per_image: int = 1,
        simclr_min_overlap_ratio: float = 0.5,
        simclr_augment: bool = True,
        dinov2_local_crop_size: Sequence[int] | None = None,
        dinov2_n_local_crops: int = 4,
        dinov2_augment: bool = True,
        seed: int | None = None,
    ) -> None:
        self.method = method.lower().strip()
        self.final_patch_size = tuple(int(v) for v in crop_size)

        spatial_config = SpatialAugmentationConfig(
            final_patch_size=self.final_patch_size,
            p_rotation=0.2 if spatial_augmentation else 0.0,
            p_scaling=0.2 if spatial_augmentation else 0.0,
        )

        # Even with random spatial augmentation disabled, this performs
        # the center crop from initial_patch_size to final_patch_size.
        self.spatial = SpatialAugmentation3D(
            spatial_config,
            enabled=True,
        )

        self.intensity = MRIAugmentation(
            enabled=intensity_augmentation,
        )

        self._worker_rng_initialized = False

        self.collator = SSLBatchCollator(
            method=self.method,
            crop_size=self.final_patch_size,
            simclr_crop_count_per_image=simclr_crop_count_per_image,
            simclr_min_overlap_ratio=simclr_min_overlap_ratio,
            simclr_augment=simclr_augment,
            dinov2_local_crop_size=dinov2_local_crop_size,
            dinov2_n_local_crops=dinov2_n_local_crops,
            dinov2_augment=dinov2_augment,
            seed=seed,
        )

    def __call__(self, **batch: Any) -> dict[str, Any]:
        # batchgenerators seeds NumPy separately in each worker.
        # Derive method-specific private RNGs once from that state.
        if not self._worker_rng_initialized:
            self.collator.reseed_from_worker()
            self._worker_rng_initialized = True

        data = np.asarray(batch["data"], dtype=np.float32)

        if data.ndim != 5:
            raise ValueError(
                "SSLBatchTransform expects (B,C,D,H,W), got "
                f"{data.shape}."
            )

        # Common nnSSL-style augmentation stage.
        data = self.spatial(data)
        data = self.intensity(data)

        data = np.ascontiguousarray(
            data,
            dtype=np.float32,
        )

        expected = self.final_patch_size
        if tuple(data.shape[2:]) != expected:
            raise RuntimeError(
                "Spatial transform returned "
                f"{tuple(data.shape[2:])}; expected {expected}."
            )

        n = int(data.shape[0])

        case_ids = batch.get("case_id", [None] * n)
        paths = batch.get("path", [None] * n)
        properties = batch.get("properties", [None] * n)

        samples = [
            {
                "data": data[i],
                "case_id": case_ids[i],
                "path": paths[i],
                "properties": properties[i],
            }
            for i in range(n)
        ]

        return self.collator(samples)
