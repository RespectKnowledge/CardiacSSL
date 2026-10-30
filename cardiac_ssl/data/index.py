from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _collect_image_paths(obj: Any, output: list[str]) -> None:
    """Recursively collect image_path entries from an nnSSL pretrain index."""
    if isinstance(obj, dict):
        image_path = obj.get("image_path")
        if isinstance(image_path, str):
            output.append(image_path)

        for value in obj.values():
            _collect_image_paths(value, output)

    elif isinstance(obj, list):
        for value in obj:
            _collect_image_paths(value, output)


def load_nnssl_case_paths(
    index_file: str | Path,
    preprocessed_root: str | Path,
) -> list[str]:
    """Load indexed nnSSL cases without filesystem traversal.

    Paths remain strings deliberately. Large foundation-model datasets can
    contain hundreds of thousands of entries, so constructing Path objects
    eagerly for every case adds unnecessary startup overhead.
    """
    index_file = Path(index_file)
    preprocessed_root = str(preprocessed_root)

    with index_file.open("r") as f:
        index = json.load(f)

    raw_paths: list[str] = []
    _collect_image_paths(
        index.get("datasets", {}),
        raw_paths,
    )

    if not raw_paths:
        raise ValueError(
            f"No image_path entries found in {index_file}"
        )

    prefix = "$nnssl_preprocessed/"

    # Avoid Path construction in this 777k-entry hot loop.
    root = preprocessed_root.rstrip("/")

    case_paths: list[str] = []

    for raw in raw_paths:
        if raw.startswith(prefix):
            path = root + "/" + raw[len(prefix):]
        else:
            path = raw

        if not path.endswith(".b2nd"):
            path += ".b2nd"

        case_paths.append(path)

    return case_paths
