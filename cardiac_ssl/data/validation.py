from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from cardiac_ssl.data.dataset import FMBenchDataset


@dataclass
class ImageRecord:
    dataset: str
    subject_id: str
    session_id: str
    image_name: str
    modality: str | None
    logical_path: str
    b2nd_path: str
    pkl_path: str


@dataclass
class ValidationResult:
    record: ImageRecord
    valid: bool
    reason: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None


def _roots() -> dict[str, Path]:
    """
    Logical roots supported by FM-Bench manifests.

    Environment variables take precedence. Standard project layouts are
    used as fallbacks for backward compatibility with nnSSL Dataset501.
    """
    repo_root = Path(__file__).resolve().parents[2]

    return {
        "$fmbench_preprocessed": Path(
            os.environ.get(
                "FMBENCH_PREPROCESSED",
                repo_root / "fmbench_preprocessed",
            )
        ).expanduser().resolve(),

        "$nnssl_preprocessed": Path(
            os.environ.get(
                "NNSSL_PREPROCESSED",
                "/projects/u6rm/my_projects/SSL/nnssl_preprocessed",
            )
        ).expanduser().resolve(),
    }


def resolve_image_path(path: str) -> Path:
    roots = _roots()

    for prefix, root in roots.items():
        if path == prefix:
            return root
        if path.startswith(prefix + "/"):
            relative = path[len(prefix) + 1 :]
            return root / relative

    return Path(path).expanduser().resolve()


def iter_image_records(collection: dict[str, Any]) -> Iterator[ImageRecord]:
    datasets = collection.get("datasets") or {}

    for dataset_key, dataset in datasets.items():
        subjects = dataset.get("subjects") or {}

        for subject_key, subject in subjects.items():
            subject_id = subject.get("subject_id") or subject_key
            sessions = subject.get("sessions") or {}

            for session_key, session in sessions.items():
                session_id = session.get("session_id") or session_key

                for image in session.get("images") or []:
                    logical = image.get("image_path")
                    if not logical:
                        continue

                    base = resolve_image_path(logical)

                    if base.suffix == ".b2nd":
                        b2nd = base
                        pkl = base.with_suffix(".pkl")
                    else:
                        b2nd = Path(str(base) + ".b2nd")
                        pkl = Path(str(base) + ".pkl")

                    yield ImageRecord(
                        dataset=str(dataset_key),
                        subject_id=str(subject_id),
                        session_id=str(session_id),
                        image_name=str(image.get("name") or base.name),
                        modality=image.get("modality"),
                        logical_path=str(logical),
                        b2nd_path=str(b2nd),
                        pkl_path=str(pkl),
                    )


def validate_record(record: ImageRecord) -> ValidationResult:
    b2nd = Path(record.b2nd_path)
    pkl = Path(record.pkl_path)

    if not b2nd.is_file():
        return ValidationResult(record, False, "missing_b2nd")

    if not pkl.is_file():
        return ValidationResult(record, False, "missing_pkl")

    try:
        with pkl.open("rb") as f:
            properties = pickle.load(f)
        if properties is None:
            return ValidationResult(record, False, "invalid_pkl")
    except Exception:
        return ValidationResult(record, False, "unreadable_pkl")

    try:
        arr = FMBenchDataset._open_blosc2(b2nd)

        shape = tuple(int(x) for x in arr.shape)
        dtype = str(arr.dtype)

        if len(shape) not in (3, 4):
            return ValidationResult(
                record, False, "invalid_shape", shape, dtype
            )

        if any(x <= 0 for x in shape):
            return ValidationResult(
                record, False, "invalid_shape", shape, dtype
            )

        # Exercise real decompression without loading the whole volume.
        if len(shape) == 4:
            spatial = shape[-3:]
            index = (
                slice(0, 1),
                slice(0, min(16, spatial[0])),
                slice(0, min(16, spatial[1])),
                slice(0, min(16, spatial[2])),
            )
        else:
            index = tuple(slice(0, min(16, x)) for x in shape)

        sample = np.asarray(arr[index])

        if sample.size == 0:
            return ValidationResult(
                record, False, "empty_data", shape, dtype
            )

        if not np.isfinite(sample).all():
            return ValidationResult(
                record, False, "nonfinite", shape, dtype
            )

        return ValidationResult(
            record=record,
            valid=True,
            shape=shape,
            dtype=dtype,
        )

    except Exception:
        return ValidationResult(record, False, "unreadable_b2nd")


def remove_invalid_images(
    collection: dict[str, Any],
    valid_logical_paths: set[str],
) -> dict[str, Any]:
    """
    Filter images while preserving the original Collection hierarchy.
    Empty sessions/subjects/datasets are removed.
    """
    clean = {
        "collection_index": collection.get("collection_index"),
        "collection_name": collection.get("collection_name"),
        "datasets": {},
    }

    for dataset_key, dataset in (collection.get("datasets") or {}).items():
        new_dataset = {
            k: v for k, v in dataset.items() if k != "subjects"
        }
        new_subjects = {}

        for subject_key, subject in (dataset.get("subjects") or {}).items():
            new_subject = {
                k: v for k, v in subject.items() if k != "sessions"
            }
            new_sessions = {}

            for session_key, session in (
                subject.get("sessions") or {}
            ).items():
                images = [
                    image
                    for image in (session.get("images") or [])
                    if image.get("image_path") in valid_logical_paths
                ]

                if images:
                    new_session = dict(session)
                    new_session["images"] = images
                    new_sessions[session_key] = new_session

            if new_sessions:
                new_subject["sessions"] = new_sessions
                new_subjects[subject_key] = new_subject

        if new_subjects:
            new_dataset["subjects"] = new_subjects
            clean["datasets"][dataset_key] = new_dataset

    return clean


def validate_collection(
    manifest: str | Path,
    output_dir: str | Path,
    progress_every: int = 10000,
) -> dict[str, Any]:
    manifest = Path(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {manifest}", flush=True)

    with manifest.open() as f:
        collection = json.load(f)

    counts: Counter[str] = Counter()
    valid_paths: set[str] = set()
    invalid: list[dict[str, Any]] = []

    seen_storage_paths: set[str] = set()
    duplicate_storage_paths = 0

    subject_ids: set[tuple[str, str]] = set()
    dataset_counts: Counter[str] = Counter()

    total = 0

    for record in iter_image_records(collection):
        total += 1
        dataset_counts[record.dataset] += 1
        subject_ids.add((record.dataset, record.subject_id))

        if record.b2nd_path in seen_storage_paths:
            duplicate_storage_paths += 1
        else:
            seen_storage_paths.add(record.b2nd_path)

        result = validate_record(record)

        if result.valid:
            counts["valid"] += 1
            valid_paths.add(record.logical_path)
        else:
            reason = result.reason or "unknown"
            counts[reason] += 1

            invalid.append(
                {
                    **asdict(record),
                    "reason": reason,
                    "shape": result.shape,
                    "dtype": result.dtype,
                }
            )

        if total % progress_every == 0:
            print(
                f"Checked {total:,} | "
                f"valid={counts['valid']:,} | "
                f"invalid={total - counts['valid']:,}",
                flush=True,
            )

    clean_collection = remove_invalid_images(
        collection,
        valid_paths,
    )

    report = {
        "collection_name": collection.get("collection_name"),
        "source_manifest": str(manifest.resolve()),
        "indexed_images": total,
        "valid_images": counts["valid"],
        "invalid_images": total - counts["valid"],
        "unique_subjects": len(subject_ids),
        "duplicate_storage_paths": duplicate_storage_paths,
        "failure_counts": {
            key: value
            for key, value in sorted(counts.items())
            if key != "valid"
        },
        "images_by_dataset": dict(sorted(dataset_counts.items())),
    }

    clean_path = output_dir / "clean_manifest.json"
    report_path = output_dir / "validation_report.json"
    invalid_path = output_dir / "invalid_images.json"

    print("Writing clean manifest...", flush=True)
    with clean_path.open("w") as f:
        json.dump(clean_collection, f)

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    with invalid_path.open("w") as f:
        json.dump(invalid, f, indent=2)

    print("\n" + "=" * 64)
    print("FM-Bench Dataset Validation")
    print("=" * 64)
    print(f"Collection:              {report['collection_name']}")
    print(f"Indexed images:          {total:,}")
    print(f"Valid images:            {counts['valid']:,}")
    print(f"Invalid images:          {total-counts['valid']:,}")
    print(f"Unique subjects:         {len(subject_ids):,}")
    print(f"Duplicate storage paths: {duplicate_storage_paths:,}")

    for reason, n in report["failure_counts"].items():
        print(f"{reason:24s}{n:,}")

    print("-" * 64)
    print(f"Clean manifest: {clean_path}")
    print(f"Report:         {report_path}")
    print(f"Invalid list:   {invalid_path}")
    print("=" * 64)

    return report
