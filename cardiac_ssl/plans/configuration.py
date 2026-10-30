"""Data planning configuration for FM-Bench.

The plan describes preprocessing only. Model, SSL-method, and optimization
parameters belong in their respective experiment configurations.

The loader is intentionally tolerant of legacy nnSSL plan files so that
existing preprocessed datasets can be used without reprocessing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration for one data representation."""

    name: str
    data_identifier: str
    spacing_style: str
    spacing: tuple[float, float, float] | None

    normalization_schemes: tuple[str, ...] = ("ZScoreNormalization",)
    use_mask_for_norm: tuple[bool, ...] = (False,)

    resampling_fn_data: str = "resample_data_or_seg_to_shape"
    resampling_fn_data_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "is_seg": False,
            "order": 3,
            "order_z": 0,
            "force_separate_z": None,
        }
    )

    resampling_fn_mask: str = "resample_data_or_seg_to_shape"
    resampling_fn_mask_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
    )

    # Training patch size is retained here because it determines how
    # preprocessed volumes are sampled by the common data pipeline.
    patch_size: tuple[int, int, int] | None = None

    @classmethod
    def from_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
    ) -> "PreprocessingConfig":
        """Create a preprocessing config from FM-Bench or legacy nnSSL data."""

        spacing = cfg.get("spacing")
        patch_size = cfg.get("patch_size")

        return cls(
            name=name,
            data_identifier=cfg["data_identifier"],
            spacing_style=cfg["spacing_style"],
            spacing=tuple(spacing) if spacing is not None else None,
            normalization_schemes=tuple(
                cfg.get("normalization_schemes", ["ZScoreNormalization"])
            ),
            use_mask_for_norm=tuple(
                cfg.get("use_mask_for_norm", [False])
            ),
            resampling_fn_data=cfg.get(
                "resampling_fn_data",
                "resample_data_or_seg_to_shape",
            ),
            resampling_fn_data_kwargs=dict(
                cfg.get("resampling_fn_data_kwargs", {})
            ),
            resampling_fn_mask=cfg.get(
                "resampling_fn_mask",
                "resample_data_or_seg_to_shape",
            ),
            resampling_fn_mask_kwargs=dict(
                cfg.get("resampling_fn_mask_kwargs", {})
            ),
            patch_size=(
                tuple(int(x) for x in patch_size)
                if patch_size is not None
                else None
            ),
        )

    def validate(self) -> None:
        if self.spacing_style not in {"median", "onemmiso", "noresample"}:
            raise ValueError(
                f"Unknown spacing_style: {self.spacing_style}"
            )

        if self.spacing_style == "noresample":
            if self.spacing is not None:
                raise ValueError(
                    "spacing must be None when spacing_style='noresample'"
                )
        else:
            if self.spacing is None:
                raise ValueError(
                    f"spacing is required for spacing_style='{self.spacing_style}'"
                )

            if len(self.spacing) != 3 or any(x <= 0 for x in self.spacing):
                raise ValueError(
                    f"Expected positive 3D spacing, got {self.spacing}"
                )

        if self.patch_size is not None:
            if len(self.patch_size) != 3 or any(x <= 0 for x in self.patch_size):
                raise ValueError(
                    f"Expected positive 3D patch_size, got {self.patch_size}"
                )


@dataclass
class DataPlan:
    """Dataset-level preprocessing plan."""

    dataset_name: str
    plans_name: str

    original_median_spacing_after_transp: tuple[float, float, float]
    transpose_forward: tuple[int, int, int]
    transpose_backward: tuple[int, int, int]

    configurations: dict[str, PreprocessingConfig]

    image_reader_writer: str | None = None
    experiment_planner_used: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataPlan":
        configurations = {
            name: PreprocessingConfig.from_dict(name, cfg)
            for name, cfg in data["configurations"].items()
        }

        plan = cls(
            dataset_name=data["dataset_name"],
            plans_name=data.get("plans_name", "fmbench_plans"),
            original_median_spacing_after_transp=tuple(
                float(x)
                for x in data["original_median_spacing_after_transp"]
            ),
            transpose_forward=tuple(
                int(x) for x in data["transpose_forward"]
            ),
            transpose_backward=tuple(
                int(x) for x in data["transpose_backward"]
            ),
            configurations=configurations,
            image_reader_writer=data.get("image_reader_writer"),
            experiment_planner_used=data.get("experiment_planner_used"),
        )

        plan.validate()
        return plan

    @classmethod
    def load(cls, path: str | Path) -> "DataPlan":
        path = Path(path)

        with path.open("r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def get_config(self, name: str) -> PreprocessingConfig:
        try:
            return self.configurations[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.configurations))
            raise KeyError(
                f"Unknown configuration '{name}'. "
                f"Available configurations: {available}"
            ) from exc

    def validate(self) -> None:
        if sorted(self.transpose_forward) != [0, 1, 2]:
            raise ValueError(
                f"Invalid transpose_forward: {self.transpose_forward}"
            )

        if sorted(self.transpose_backward) != [0, 1, 2]:
            raise ValueError(
                f"Invalid transpose_backward: {self.transpose_backward}"
            )

        if len(self.original_median_spacing_after_transp) != 3:
            raise ValueError(
                "original_median_spacing_after_transp must contain 3 values"
            )

        for config in self.configurations.values():
            config.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_plan(path: str | Path) -> DataPlan:
    """Load an FM-Bench or legacy nnSSL preprocessing plan."""
    return DataPlan.load(path)
