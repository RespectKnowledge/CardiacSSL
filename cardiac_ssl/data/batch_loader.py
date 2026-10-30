"""nnSSL-style patch batch generation for FM-Bench."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader

from cardiac_ssl.data.dataset import FMBenchDataset
from cardiac_ssl.data.patch_loader import load_patch


class FMBenchPatchDataLoader3D(DataLoader):
    """Generate random 3D patch batches directly from Blosc2 volumes.

    Cases are sampled by batchgenerators. Only the requested patch region
    is materialized from each memory-mapped Blosc2 volume.
    """

    def __init__(
        self,
        dataset: FMBenchDataset,
        batch_size: int,
        patch_size: Sequence[int],
        final_patch_size: Sequence[int],
        seed: int | None = None,
    ):
        super().__init__(
            dataset,
            batch_size,
            1,
            None,
            True,
            False,
            True,
            None,
        )

        self.dataset = dataset
        self.indices = list(range(len(dataset)))

        self.patch_size = tuple(int(x) for x in patch_size)
        self.final_patch_size = tuple(int(x) for x in final_patch_size)

        if len(self.patch_size) != 3:
            raise ValueError("patch_size must contain 3 dimensions.")

        if len(self.final_patch_size) != 3:
            raise ValueError("final_patch_size must contain 3 dimensions.")

        if any(
            p < f
            for p, f in zip(self.patch_size, self.final_patch_size)
        ):
            raise ValueError(
                "patch_size must be >= final_patch_size "
                "in every dimension."
            )

        # Base seed is stored so each multiprocessing worker can create
        # an independent Generator after fork.
        self.base_seed = (
            int(seed)
            if seed is not None
            else int(np.random.SeedSequence().entropy)
        )
        self.rng = np.random.default_rng(self.base_seed)

        # Determine channel count from Blosc2 metadata only.
        first_path = dataset.get_case_path(0)
        arr = dataset._open_blosc2(first_path)

        if len(arr.shape) != 4:
            raise ValueError(
                f"Expected (C,D,H,W), got {tuple(arr.shape)}"
            )

        self.num_channels = int(arr.shape[0])

        self.data_shape = (
            batch_size,
            self.num_channels,
            *self.patch_size,
        )

    def set_thread_id(self, thread_id: int) -> None:
        """Initialize independent RNG state inside each worker process."""
        super().set_thread_id(thread_id)

        self.rng = np.random.default_rng(
            self.base_seed + int(thread_id)
        )

    def generate_train_batch(self):
        selected_indices = self.get_indices()

        data_all = np.zeros(
            self.data_shape,
            dtype=np.float32,
        )

        case_ids = []
        paths = []
        bboxes = []

        for j, index in enumerate(selected_indices):
            index = int(index)

            path = self.dataset.get_case_path(index)

            patch, bbox = load_patch(
                image_path=path,
                patch_size=self.patch_size,
                final_patch_size=self.final_patch_size,
                rng=self.rng,
            )

            data_all[j] = patch

            case_ids.append(self.dataset.get_case_id(index))
            paths.append(str(path))
            bboxes.append(bbox)

        return {
            "data": data_all,
            "case_id": case_ids,
            "path": paths,
            "bbox": bboxes,
        }
