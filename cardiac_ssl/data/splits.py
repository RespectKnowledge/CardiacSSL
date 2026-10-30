"""Subject-level train/validation splitting for FM-Bench."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


def create_subject_split(
    subject_ids: Iterable[str],
    *,
    seed: int = 12345,
    max_val_subjects: int = 1000,
) -> dict[str, list[str]]:
    """Create the deterministic subject split used by nnSSL."""
    subjects = sorted(set(str(x) for x in subject_ids))

    if not subjects:
        raise ValueError("Cannot create a split from zero subjects.")

    n_val = min(
        int(max_val_subjects),
        max(int(len(subjects) / 100), 5),
    )
    n_val = min(n_val, len(subjects))

    rng = random.Random(seed)
    val = rng.sample(subjects, n_val)
    val_set = set(val)
    train = [x for x in subjects if x not in val_set]

    return {"train": train, "val": val}


def validate_subject_split(
    split: dict[str, list[str]],
) -> None:
    if "train" not in split or "val" not in split:
        raise ValueError("Split must contain 'train' and 'val'.")

    train = split["train"]
    val = split["val"]

    if len(train) != len(set(train)):
        raise ValueError("Duplicate subjects found in train split.")

    if len(val) != len(set(val)):
        raise ValueError("Duplicate subjects found in validation split.")

    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(
            f"Train/validation subject leakage: {len(overlap)} subjects."
        )


def save_subject_split(
    split: dict[str, list[str]],
    path: str | Path,
) -> Path:
    validate_subject_split(split)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(split, f, indent=2)

    return path


def load_subject_split(
    path: str | Path,
) -> dict[str, list[str]]:
    path = Path(path)

    with path.open() as f:
        split = json.load(f)

    validate_subject_split(split)
    return split
