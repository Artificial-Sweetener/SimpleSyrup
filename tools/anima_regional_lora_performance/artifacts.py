# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve and verify pinned Anima performance artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .manifest import PerformanceArtifact


class PerformanceArtifactManifest(Protocol):
    """Expose one immutable collection of performance artifacts."""

    @property
    def artifacts(self) -> tuple[PerformanceArtifact, ...]:
        """Return exact benchmark artifacts in declared order."""

        ...


def require_artifact(
    manifest: PerformanceArtifactManifest,
    role: str,
) -> PerformanceArtifact:
    """Return the uniquely declared artifact for one benchmark role."""

    matches = tuple(
        artifact for artifact in manifest.artifacts if artifact.role == role
    )
    if len(matches) != 1:
        raise ValueError(f"Anima performance requires exactly one {role!r} artifact.")
    return matches[0]


def verify_artifact(artifact: PerformanceArtifact) -> None:
    """Require exact local size and SHA-256 before model loading."""

    if not artifact.path.is_file():
        raise FileNotFoundError(
            f"Anima performance artifact is missing: {artifact.path}"
        )
    if artifact.path.stat().st_size != artifact.size_bytes:
        raise ValueError(f"Anima performance artifact size changed: {artifact.path}")
    if _file_sha256(artifact.path) != artifact.sha256:
        raise ValueError(f"Anima performance artifact hash changed: {artifact.path}")


def _file_sha256(path: Path) -> str:
    """Hash one large artifact through bounded reads."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
