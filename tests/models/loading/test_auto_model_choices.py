# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for artifact-aware automatic component dropdown choices."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import pytest

from simple_syrup.runtime.auto_model_artifact import AutoModelArtifact
from simple_syrup.runtime.auto_model_choices import automatic_component_choices


class _FolderPaths(ModuleType):
    """Resolve ComfyUI-relative test model choices from one models root."""

    def __init__(self, models_root: Path) -> None:
        """Create a model-path fake rooted at the temporary directory."""

        super().__init__("folder_paths")
        self._models_root = models_root

    def get_full_path(self, folder_name: str, choice: str) -> str | None:
        """Return an existing test model path or None."""

        path = self._models_root / folder_name / choice
        return str(path) if path.is_file() else None


def test_choices_hide_official_and_renamed_automatic_files(tmp_path: Path) -> None:
    """Known local artifacts appear only through their automatic leading choice."""

    folder_paths = _FolderPaths(tmp_path / "models")
    artifact = _artifact("official.safetensors", b"trusted")
    folder = tmp_path / "models" / "text_encoders"
    folder.mkdir(parents=True)
    (folder / "renamed.safetensors").write_bytes(b"trusted")
    (folder / "manual.safetensors").write_bytes(b"manual model")

    choices = automatic_component_choices(
        installed=(
            artifact.filename,
            "renamed.safetensors",
            "manual.safetensors",
        ),
        artifacts=(artifact,),
        leading_choices=("auto", artifact.filename),
        folder_paths_module=folder_paths,
    )

    assert choices == ["auto", artifact.filename, "manual.safetensors"]


def test_choices_keep_same_category_files_with_different_sizes(tmp_path: Path) -> None:
    """Manual files remain selectable when they cannot be the automatic artifact."""

    folder_paths = _FolderPaths(tmp_path / "models")
    artifact = _artifact("official.safetensors", b"trusted")
    folder = tmp_path / "models" / "text_encoders"
    folder.mkdir(parents=True)
    (folder / "custom.safetensors").write_bytes(b"different size")

    choices = automatic_component_choices(
        installed=("custom.safetensors",),
        artifacts=(artifact,),
        leading_choices=("auto",),
        folder_paths_module=folder_paths,
    )

    assert choices == ["auto", "custom.safetensors"]


def test_choices_require_artifacts_from_one_model_category() -> None:
    """Dropdown filtering cannot accidentally combine unrelated model categories."""

    folder_paths = _FolderPaths(Path("models"))

    with pytest.raises(ValueError, match="share one model category"):
        automatic_component_choices(
            installed=(),
            artifacts=(
                _artifact("encoder.safetensors", b"encoder", "text_encoders"),
                _artifact("vae.safetensors", b"vae", "vae"),
            ),
            leading_choices=("auto",),
            folder_paths_module=folder_paths,
        )


def _artifact(
    filename: str,
    content: bytes,
    folder_name: str = "text_encoders",
) -> AutoModelArtifact:
    """Create a small trusted artifact for dropdown tests."""

    return AutoModelArtifact(
        cache_id=filename,
        filename=filename,
        folder_name=folder_name,
        canonical_subfolder="test",
        source_url=f"https://example.invalid/{filename}",
        source_repo="example/models",
        description=filename,
        sha256=hashlib.sha256(content).hexdigest(),
        file_size_bytes=len(content),
    )
