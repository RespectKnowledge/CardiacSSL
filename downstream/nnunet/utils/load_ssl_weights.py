"""
Utilities for transferring Cardiac-SSL Swin3D weights into the
downstream Swin segmentation network.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _extract_state_dict(checkpoint):
    """
    Accept common SSL checkpoint layouts.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary.")

    # Standard Cardiac-SSL trainer checkpoint.
    if "model_state" in checkpoint and isinstance(
        checkpoint["model_state"], dict
    ):
        return checkpoint["model_state"]

    # Prefer explicitly exported encoder weights.
    for key in (
        "encoder_state_dict",
        "backbone_state_dict",
        "encoder",
        "backbone",
    ):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # Raw state_dict.
    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint

    raise RuntimeError(
        "Could not identify encoder state_dict in checkpoint."
    )


def load_ssl_encoder(
    network,
    checkpoint_path,
    strict=True,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = _extract_state_dict(checkpoint)

    # Remove common wrappers while preserving the internal Swin names.
    cleaned = {}

    # Convert SSL model names to the names expected by
    # network.encoder.state_dict().
    #
    # Cardiac-SSL checkpoints:
    #   backbone.encoder.layers1...
    #
    # Downstream encoder target:
    #   encoder.layers1...
    #
    # Therefore remove backbone. but PRESERVE encoder.
    prefixes = (
        "module.",
        "student.",
        "teacher.",
        "context_encoder.",
        "backbone.",
    )

    for key, value in state.items():
        new_key = key

        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    target = network.encoder.state_dict()

    matched = {}
    shape_mismatch = {}
    unexpected = []

    for key, value in cleaned.items():
        if key not in target:
            unexpected.append(key)
            continue

        if target[key].shape != value.shape:
            shape_mismatch[key] = (
                tuple(value.shape),
                tuple(target[key].shape),
            )
            continue

        matched[key] = value

    missing = sorted(set(target) - set(matched))

    print("=" * 72)
    print("Cardiac SSL -> downstream Swin encoder transfer")
    print("=" * 72)
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Target tensors:   {len(target)}")
    print(f"Matched tensors:  {len(matched)}")
    print(f"Missing tensors:  {len(missing)}")
    print(f"Shape mismatch:   {len(shape_mismatch)}")
    print(f"Unexpected:       {len(unexpected)}")

    if shape_mismatch:
        print("\nShape mismatches:")
        for k, v in shape_mismatch.items():
            print(f"  {k}: checkpoint={v[0]}, target={v[1]}")

    if missing:
        print("\nMissing target tensors:")
        for k in missing[:30]:
            print(" ", k)

    if strict and (missing or shape_mismatch):
        raise RuntimeError(
            "SSL encoder is not fully compatible with downstream Swin."
        )

    network.encoder.load_state_dict(
        matched,
        strict=False,
    )

    print("\nEncoder weight transfer: PASS")

    return {
        "matched": len(matched),
        "target": len(target),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "unexpected": unexpected,
    }
