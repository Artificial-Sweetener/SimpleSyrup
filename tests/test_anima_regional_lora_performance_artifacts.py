# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify pinned Anima performance artifact role and byte identity."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from tools.anima_regional_lora_performance.artifacts import (
    require_artifact,
    verify_artifact,
)
from tools.anima_regional_lora_performance.manifest import (
    PerformanceArtifact,
    default_manifest,
)


def test_artifact_verification_requires_exact_bytes(tmp_path: Path) -> None:
    """Reject missing, resized, or content-changed performance artifacts."""

    path = tmp_path / "fixture.safetensors"
    value = b"regional-lora"
    path.write_bytes(value)
    artifact = PerformanceArtifact(
        "fixture",
        path,
        len(value),
        hashlib.sha256(value).hexdigest(),
    )

    verify_artifact(artifact)
    with pytest.raises(ValueError, match="size changed"):
        verify_artifact(replace(artifact, size_bytes=len(value) + 1))
    with pytest.raises(ValueError, match="hash changed"):
        verify_artifact(replace(artifact, sha256="0" * 64))
    with pytest.raises(FileNotFoundError, match="artifact is missing"):
        verify_artifact(replace(artifact, path=tmp_path / "missing"))


def test_artifact_roles_are_unique_and_required() -> None:
    """Prevent ambiguous model or adapter selection before loading."""

    first = PerformanceArtifact("model", Path("model.safetensors"), 1, "0" * 64)
    second = PerformanceArtifact(
        "regional_lora", Path("adapter.safetensors"), 1, "1" * 64
    )
    manifest = default_manifest(artifacts=(first, second))

    assert require_artifact(manifest, "regional_lora").role == "regional_lora"
    with pytest.raises(ValueError, match="exactly one 'missing'"):
        require_artifact(manifest, "missing")
    with pytest.raises(ValueError, match="exactly one 'regional_lora'"):
        require_artifact(
            replace(
                manifest,
                artifacts=(*manifest.artifacts, manifest.artifacts[1]),
            ),
            "regional_lora",
        )
