# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify externally inventoried model visibility for the Anima oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.anima_regression_oracle.model_visibility import (
    ManagedOracleModelVisibility,
)


def test_inventory_requires_exact_declared_alias_coverage(tmp_path: Path) -> None:
    """Reject missing aliases without depending on any installed model identity."""

    inventory = _inventory(tmp_path, ("family/a.safetensors",))

    with pytest.raises(ValueError, match="exactly cover"):
        ManagedOracleModelVisibility.from_inventory(
            inventory,
            model_root=_model_root(tmp_path),
            required_selections=("family/a.safetensors", "family/b.safetensors"),
        )


def test_inventory_creates_and_cleans_all_declared_aliases(tmp_path: Path) -> None:
    """Expose every declared alias transactionally from anonymous sources."""

    selections = ("family/a.safetensors", "family/b.safetensors")
    inventory = _inventory(tmp_path, selections)
    model_root = _model_root(tmp_path)
    owner = ManagedOracleModelVisibility.from_inventory(
        inventory,
        model_root=model_root,
        required_selections=selections,
    )

    with owner:
        assert all(
            (model_root / "loras" / selection).is_file() for selection in selections
        )

    assert owner.cleaned
    assert all(
        not (model_root / "loras" / selection).exists() for selection in selections
    )


def _model_root(root: Path) -> Path:
    """Create one minimal Comfy model root."""

    model_root = root / "models"
    (model_root / "loras").mkdir(parents=True)
    return model_root


def _inventory(root: Path, selections: tuple[str, ...]) -> Path:
    """Write one anonymous exact-hash visibility inventory."""

    entries: list[dict[str, object]] = []
    for index, selection in enumerate(selections):
        source = root / f"source-{index}.safetensors"
        payload = f"source-{index}".encode()
        source.write_bytes(payload)
        entries.append(
            {
                "source": str(source),
                "category": "loras",
                "selection_name": selection,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    path = root / "inventory.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path
