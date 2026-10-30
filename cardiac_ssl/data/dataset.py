"""Preprocessed 3D MRI dataset for FM-Bench.

Supports existing nnSSL-style Blosc2 preprocessing so large datasets can be
used directly without reprocessing.

Each case consists of:
    <case>.b2nd   preprocessed image
    <case>.pkl    preprocessing properties, when available

The dataset intentionally has no dependency on nnSSL.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import blosc2
import numpy as np
from torch.utils.data import Dataset


class FMBenchDataset(Dataset):
    """Dataset of preprocessed 3D MRI volumes stored as Blosc2 arrays."""

    def __init__(
        self,
        root: str | Path,
        case_paths: Sequence[str | Path] | None = None,
    ):
        self.root = Path(root).expanduser().resolve()

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Preprocessed directory does not exist: {self.root}"
            )

        if case_paths is None:
            # Generic fallback for datasets without an explicit index.
            self.case_paths = sorted(self.root.rglob("*.b2nd"))
        else:
            # Indexed datasets already provide the complete inventory.
            # Keep entries in their lightweight representation and convert
            # only sampled cases to Path objects on demand. This avoids a
            # second O(N) Path-construction pass for very large datasets.
            self.case_paths = case_paths

        if not self.case_paths:
            raise RuntimeError(
                f"No .b2nd volumes found under {self.root}"
            )

    def _normalise_case_path(self, path: str | Path) -> Path:
        """Normalise an indexed case path without filesystem access."""
        path = Path(path).expanduser()

        if not path.is_absolute():
            path = self.root / path

        if path.suffix != ".b2nd":
            raise ValueError(
                f"Expected a .b2nd image, got: {path}"
            )

        return path

    def _validate_case_path(self, path: Path) -> Path:
        """Validate a case lazily when that case is actually accessed."""
        if not path.is_file():
            raise FileNotFoundError(
                f"Indexed .b2nd volume does not exist: {path}"
            )

        return path

    def __len__(self) -> int:
        return len(self.case_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.load_case(index)

    def get_case_path(self, index: int) -> Path:
        path = self.case_paths[index]

        if isinstance(path, Path):
            return path

        return self._normalise_case_path(path)

    def get_case_id(self, index: int) -> str:
        """Return a stable case ID across equivalent HPC mount aliases."""
        path = self.get_case_path(index)

        try:
            relative = path.relative_to(self.root)
        except ValueError:
            # The same filesystem can be exposed through different aliases,
            # for example /projects/... and /lus/lfs1aip2/projects/....
            # Avoid Path.resolve() here because this method is on the
            # training hot path.
            root_name = self.root.name
            positions = [
                i
                for i, part in enumerate(path.parts)
                if part == root_name
            ]

            if not positions:
                return path.stem

            relative = Path(
                *path.parts[positions[-1] + 1:]
            )

        return str(relative.with_suffix(""))

    @staticmethod
    def _properties_path(image_path: Path) -> Path:
        return image_path.with_suffix(".pkl")

    @staticmethod
    def _open_blosc2(path: Path):
        """Open a Blosc2 volume lazily for patch-wise access.

        FM-Bench follows the nnSSL loading strategy here: one Blosc thread
        per worker and memory-mapped read access. The full MRI is not
        materialized at dataset access time.
        """
        return blosc2.open(
            urlpath=str(path),
            mode="r",
            dparams={"nthreads": 1},
            mmap_mode="r",
        )

    @staticmethod
    def _load_blosc2(path: Path) -> np.ndarray:
        """Materialize a complete volume when explicitly required.

        Training should prefer _open_blosc2 and slice the requested patch
        directly from the lazy Blosc2 array.
        """
        arr = FMBenchDataset._open_blosc2(path)
        data = np.asarray(arr[:])

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(
                f"Expected numeric MRI data, got {data.dtype} in {path}"
            )

        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"NaN or Inf found in preprocessed volume: {path}"
            )

        return data

    @staticmethod
    def _load_properties(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}

        with path.open("rb") as f:
            properties = pickle.load(f)

        if not isinstance(properties, dict):
            raise TypeError(
                f"Expected properties dict in {path}, "
                f"got {type(properties).__name__}"
            )

        return properties

    def load_case(self, index: int) -> dict[str, Any]:
        image_path = self._validate_case_path(
            self.get_case_path(index)
        )

        data = self._load_blosc2(image_path)
        properties = self._load_properties(
            self._properties_path(image_path)
        )

        # FM-Bench convention:
        # image tensors/arrays are always channel-first: (C, D, H, W).
        if data.ndim == 3:
            data = data[None]

        if data.ndim != 4:
            raise ValueError(
                f"Expected 3D MRI as (C,D,H,W), got "
                f"shape {data.shape} for {image_path}"
            )

        return {
            "data": data,
            "properties": properties,
            "case_id": self.get_case_id(index),
            "path": str(image_path),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "num_cases": len(self),
            "first_case": (
                str(self.case_paths[0]) if self.case_paths else None
            ),
        }
