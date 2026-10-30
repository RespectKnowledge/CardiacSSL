"""Plan-driven data pipeline construction for FM-Bench."""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any


from cardiac_ssl.data.dataset import FMBenchDataset
from cardiac_ssl.data.index import load_nnssl_case_paths
from cardiac_ssl.data.ssl_dataset import (
    SSLTrainingDataset,
    SSLBatchTransform,
)
from cardiac_ssl.data.batch_loader import FMBenchPatchDataLoader3D
from cardiac_ssl.data.async_loader import LimitedLenWrapper
from cardiac_ssl.transforms.spatial import (
    SpatialAugmentationConfig,
    compute_initial_patch_size,
)
from cardiac_ssl.plans.configuration import (
    DataPlan,
    PreprocessingConfig,
    load_plan,
)


def resolve_preprocessing_config(
    plan: DataPlan,
    configuration: str,
) -> PreprocessingConfig:
    """Resolve and validate a training configuration from a data plan."""
    cfg = plan.get_config(configuration)

    # Legacy nnSSL preprocessing plans may not define a training
    # patch size. This is valid: build_split_loader can receive the
    # training crop via final_patch_size_override.
    return cfg


def _find_nnssl_index(
    dataset_root: str | Path,
    configuration: str,
) -> Path | None:
    """
    Locate the nnSSL pretraining index without recursively scanning data.

    Expected layout:
      <preprocessed>/<dataset>/<data_identifier>/<dataset>
      <preprocessed>/<dataset>/pretrain_data__<configuration>.json
    """
    root = Path(dataset_root)

    # dataset_root normally points to:
    # dataset / data_identifier / dataset
    if len(root.parents) >= 2:
        dataset_dir = root.parents[1]
        candidate = (
            dataset_dir
            / f"pretrain_data__{configuration}.json"
        )
        if candidate.is_file():
            return candidate

    return None


def _infer_preprocessed_root(
    dataset_root: str | Path,
) -> Path:
    """
    Infer $nnssl_preprocessed from the standard nnSSL directory layout.

    dataset_root:
      <preprocessed>/<dataset>/<data_identifier>/<dataset>
    """
    root = Path(dataset_root)

    if len(root.parents) < 3:
        raise ValueError(
            "Cannot infer nnSSL preprocessed root from "
            f"{root}"
        )

    return root.parents[2]


def build_training_dataset(
    dataset_root: str | Path,
    plan: DataPlan,
    configuration: str,
    *,
    spatial_augmentation: bool = True,
    intensity_augmentation: bool = True,
    seed: int | None = None,
) -> SSLTrainingDataset:
    """Build the common SSL dataset directly from a DataPlan."""

    cfg = resolve_preprocessing_config(
        plan,
        configuration,
    )

    index_file = _find_nnssl_index(
        dataset_root,
        configuration,
    )

    if index_file is not None:
        case_paths = load_nnssl_case_paths(
            index_file=index_file,
            preprocessed_root=_infer_preprocessed_root(
                dataset_root
            ),
        )

        base_dataset = FMBenchDataset(
            root=dataset_root,
            case_paths=case_paths,
        )
    else:
        # Generic fallback for datasets without an nnSSL index.
        # This may recursively discover .b2nd cases.
        base_dataset = FMBenchDataset(
            root=dataset_root,
        )

    return SSLTrainingDataset(
        dataset=base_dataset,
        initial_patch_size=None,
        final_patch_size=cfg.patch_size,
        spatial_augmentation=spatial_augmentation,
        intensity_augmentation=intensity_augmentation,
        seed=seed,
    )




def load_primary_split_paths(
    split_file: str | Path,
    split: str,
) -> list[str]:
    """Load train/val image paths from a frozen FM-Bench split."""

    if split not in {"train", "val"}:
        raise ValueError(
            f"split must be 'train' or 'val', got {split!r}"
        )

    split_file = Path(split_file)

    with split_file.open("r", encoding="utf-8") as f:
        splits = json.load(f)

    if not isinstance(splits, list) or len(splits) != 1:
        raise ValueError(
            "Expected splits_final.json to contain exactly one split."
        )

    split_data = splits[0]

    paths = split_data.get(split)

    if not isinstance(paths, list) or not paths:
        raise ValueError(
            f"No image paths found for split {split!r}"
        )

    nnssl_root = os.environ.get("nnssl_preprocessed")
    if not nnssl_root:
        raise RuntimeError(
            "nnssl_preprocessed environment variable is not set."
        )
    fmbench_root = (
        "/projects/u6rm/my_projects/SSL/FM-Bench/"
        "fmbench_preprocessed"
    )

    resolved: list[str] = []

    for raw in paths:
        if not isinstance(raw, str):
            raise TypeError(
                f"Expected string image path, got {type(raw).__name__}"
            )

        if raw.startswith("$nnssl_preprocessed/"):
            path = (
                nnssl_root
                + "/"
                + raw[len("$nnssl_preprocessed/"):]
            )
        elif raw.startswith("$fmbench_preprocessed/"):
            path = (
                fmbench_root
                + "/"
                + raw[len("$fmbench_preprocessed/"):]
            )
        else:
            path = raw

        if not path.endswith(".b2nd"):
            path += ".b2nd"

        resolved.append(path)

    return resolved


def build_split_loader(
    split_file: str | Path,
    split: str,
    dataset_root: str | Path,
    plan: DataPlan | str | Path,
    configuration: str,
    method: str,
    *,
    batch_size: int = 2,
    num_workers: int = 8,
    num_cached: int = 2,
    iterations_per_epoch: int = 250,
    spatial_augmentation: bool = True,
    intensity_augmentation: bool = True,
    pin_memory: bool = True,
    seed: int | None = None,
    dinov2_local_crop_size: tuple[int, int, int] | None = None,
    dinov2_n_local_crops: int = 4,
    method_augment: bool = True,
    final_patch_size_override: Sequence[int] | None = None,
):
    """Build loader directly from frozen Experiment-1 train/val paths."""

    if not isinstance(plan, DataPlan):
        plan = load_plan(plan)

    cfg = resolve_preprocessing_config(
        plan,
        configuration,
    )

    if final_patch_size_override is not None:
        final_patch_size = tuple(
            int(v) for v in final_patch_size_override
        )
    else:
        assert cfg.patch_size is not None
        final_patch_size = tuple(
            int(v) for v in cfg.patch_size
        )

    if len(final_patch_size) != 3:
        raise ValueError(
            "final_patch_size must contain exactly 3 dimensions."
        )

    spatial_config = SpatialAugmentationConfig(
        final_patch_size=final_patch_size,
        p_rotation=0.2 if spatial_augmentation else 0.0,
        p_scaling=0.2 if spatial_augmentation else 0.0,
    )

    if spatial_augmentation:
        initial_patch_size = compute_initial_patch_size(
            final_patch_size,
            spatial_config.rotation_x,
            spatial_config.rotation_y,
            spatial_config.rotation_z,
            spatial_config.scale_range,
        )
    else:
        initial_patch_size = final_patch_size

    case_paths = load_primary_split_paths(
        split_file,
        split,
    )

    base_dataset = FMBenchDataset(
        root=dataset_root,
        case_paths=case_paths,
    )

    patch_loader = FMBenchPatchDataLoader3D(
        dataset=base_dataset,
        batch_size=batch_size,
        patch_size=initial_patch_size,
        final_patch_size=final_patch_size,
        seed=seed,
    )

    transform = SSLBatchTransform(
        method=method,
        crop_size=final_patch_size,
        spatial_augmentation=spatial_augmentation,
        intensity_augmentation=intensity_augmentation,
        simclr_augment=method_augment,
        dinov2_local_crop_size=dinov2_local_crop_size,
        dinov2_n_local_crops=dinov2_n_local_crops,
        dinov2_augment=method_augment,
        seed=seed,
    )

    if num_workers <= 0:
        raise ValueError(
            "Optimized FM-Bench loader requires num_workers > 0."
        )

    worker_seeds = None
    if seed is not None:
        worker_seeds = [
            int(seed) + i
            for i in range(num_workers)
        ]

    return LimitedLenWrapper(
        iterations_per_epoch,
        patch_loader,
        transform,
        num_processes=num_workers,
        num_cached=num_cached,
        seeds=worker_seeds,
        pin_memory=pin_memory,
        wait_time=0.02,
    )


def build_train_loader(
    dataset_root: str | Path,
    plan: DataPlan | str | Path,
    configuration: str,
    method: str,
    *,
    batch_size: int = 2,
    num_workers: int = 8,
    num_cached: int = 2,
    iterations_per_epoch: int = 250,
    spatial_augmentation: bool = True,
    intensity_augmentation: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int | None = None,
    simclr_crop_count_per_image: int = 1,
    simclr_min_overlap_ratio: float = 0.5,
    dinov2_local_crop_size: tuple[int, int, int] | None = None,
    dinov2_n_local_crops: int = 4,
    method_augment: bool = True,
    **loader_kwargs: Any,
):
    """Build the optimized plan-driven FM-Bench training stream.

    The data plan determines the final patch size. An enlarged initial
    patch is computed automatically when spatial augmentation is active.
    Only that patch is read from each memory-mapped Blosc2 volume.
    """

    if not isinstance(plan, DataPlan):
        plan = load_plan(plan)

    cfg = resolve_preprocessing_config(
        plan,
        configuration,
    )

    assert cfg.patch_size is not None

    final_patch_size = tuple(
        int(v) for v in cfg.patch_size
    )

    spatial_config = SpatialAugmentationConfig(
        final_patch_size=final_patch_size,
        p_rotation=0.2 if spatial_augmentation else 0.0,
        p_scaling=0.2 if spatial_augmentation else 0.0,
    )

    if spatial_augmentation:
        initial_patch_size = compute_initial_patch_size(
            final_patch_size,
            spatial_config.rotation_x,
            spatial_config.rotation_y,
            spatial_config.rotation_z,
            spatial_config.scale_range,
        )
    else:
        initial_patch_size = final_patch_size

    # Build only the indexed base dataset. Do not wrap it in
    # SSLTrainingDataset because that path materializes complete cases.
    index_file = _find_nnssl_index(
        dataset_root,
        configuration,
    )

    if index_file is not None:
        case_paths = load_nnssl_case_paths(
            index_file=index_file,
            preprocessed_root=_infer_preprocessed_root(
                dataset_root
            ),
        )

        base_dataset = FMBenchDataset(
            root=dataset_root,
            case_paths=case_paths,
        )
    else:
        base_dataset = FMBenchDataset(
            root=dataset_root,
        )

    patch_loader = FMBenchPatchDataLoader3D(
        dataset=base_dataset,
        batch_size=batch_size,
        patch_size=initial_patch_size,
        final_patch_size=final_patch_size,
        seed=seed,
    )

    transform = SSLBatchTransform(
        method=method,
        crop_size=final_patch_size,
        spatial_augmentation=spatial_augmentation,
        intensity_augmentation=intensity_augmentation,
        simclr_crop_count_per_image=(
            simclr_crop_count_per_image
        ),
        simclr_min_overlap_ratio=(
            simclr_min_overlap_ratio
        ),
        simclr_augment=method_augment,
        dinov2_local_crop_size=(
            dinov2_local_crop_size
        ),
        dinov2_n_local_crops=(
            dinov2_n_local_crops
        ),
        dinov2_augment=method_augment,
        seed=seed,
    )

    if num_workers <= 0:
        raise ValueError(
            "Optimized FM-Bench loader requires num_workers > 0."
        )

    worker_seeds = None
    if seed is not None:
        worker_seeds = [
            int(seed) + i
            for i in range(num_workers)
        ]

    return LimitedLenWrapper(
        iterations_per_epoch,
        patch_loader,
        transform,
        num_processes=num_workers,
        num_cached=num_cached,
        seeds=worker_seeds,
        pin_memory=pin_memory,
        wait_time=0.02,
    )
