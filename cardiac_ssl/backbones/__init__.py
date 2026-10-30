"""Swin3D backbone registry for Cardiac-SSL."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .swin3d import Swin3D, Swin3DConfig


BACKBONE_NAMES = ("swin3d",)


def _as_dict(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("Backbone config must be a mapping or None.")
    return dict(config)


def build_backbone(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> nn.Module:

    name = name.lower().strip()
    cfg = _as_dict(config)

    if "params" in cfg:
        cfg = dict(cfg["params"])

    cfg.pop("name", None)

    if name == "swin3d":
        return Swin3D(Swin3DConfig(**cfg))

    raise ValueError(
        f"Unknown backbone {name!r}. "
        f"Available: {', '.join(BACKBONE_NAMES)}"
    )


__all__ = [
    "BACKBONE_NAMES",
    "build_backbone",
    "Swin3D",
    "Swin3DConfig",
]
