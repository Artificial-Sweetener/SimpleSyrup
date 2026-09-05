# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own one temporary normal-install checkpoint hard link for integration."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from tools.comfy_integration.selection_path import comfy_selection_path


@dataclass(frozen=True, slots=True)
class CheckpointArtifactIdentity:
    """Describe one private source checkpoint through public stable identity."""

    stable_name: str
    size: int
    sha256: str


class ManagedCheckpointLink:
    """Expose one validated checkpoint under Comfy's normal model directory."""

    _DIRECTORY_NAME = "simple_syrup_p8_5_sdxl"

    def __init__(
        self,
        *,
        source: Path,
        source_checkpoint_name: str,
        identity: CheckpointArtifactIdentity,
    ) -> None:
        """Retain exact link inputs without changing filesystem state."""

        self._source = source.resolve()
        try:
            self._source_checkpoint_name = comfy_selection_path(source_checkpoint_name)
        except ValueError:
            raise ValueError(
                "Source checkpoint name must be a safe relative Comfy selection."
            ) from None
        self._identity = identity
        self._model_root = self._derive_model_root()
        self._directory = self._model_root / self._DIRECTORY_NAME
        self._target = self._directory / identity.stable_name
        self._active = False
        self.cleaned = False

    @property
    def checkpoint_name(self) -> str:
        """Return Comfy's stable relative checkpoint selection value."""

        return f"{self._DIRECTORY_NAME}\\{self._identity.stable_name}"

    def __enter__(self) -> ManagedCheckpointLink:
        """Validate the source and create exactly one owned hard link."""

        self._validate()
        if self._target.exists():
            raise FileExistsError(
                f"Owned SDXL integration checkpoint link already exists: {self._target}"
            )
        created_directory = False
        if not self._directory.exists():
            self._directory.mkdir(parents=False)
            created_directory = True
        try:
            os.link(self._source, self._target)
        except BaseException:
            if created_directory and not any(self._directory.iterdir()):
                self._directory.rmdir()
            raise
        self._active = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove only the exact link and empty directory owned by this context."""

        del exc_type, exc, traceback
        if self._active:
            self._target.unlink(missing_ok=False)
            self._active = False
        if self._directory.is_dir() and not any(self._directory.iterdir()):
            self._directory.rmdir()
        self.cleaned = not self._target.exists()

    def _validate(self) -> None:
        """Fail closed on path, stable identity, size, or digest mismatch."""

        if not self._source.is_file():
            raise FileNotFoundError(
                "SDXL integration checkpoint source does not exist."
            )
        if not self._model_root.is_dir():
            raise FileNotFoundError("Comfy checkpoint directory does not exist.")
        if (self._model_root / self._source_checkpoint_name).resolve() != self._source:
            raise ValueError(
                "Checkpoint source does not match its normal Comfy selection name."
            )
        if not self._identity.stable_name or Path(self._identity.stable_name).name != (
            self._identity.stable_name
        ):
            raise ValueError("Checkpoint stable name must be one filename.")
        if self._source.stat().st_size != self._identity.size:
            raise ValueError(
                "SDXL integration checkpoint size does not match identity."
            )
        if _sha256(self._source) != self._identity.sha256.lower():
            raise ValueError("SDXL integration checkpoint SHA-256 does not match.")

    def _derive_model_root(self) -> Path:
        """Derive the configured checkpoint root from a trusted relative name."""

        relative = self._source_checkpoint_name
        if relative.name != self._identity.stable_name:
            raise ValueError(
                "Source checkpoint name must be a safe relative Comfy selection."
            )
        return self._source.parents[len(relative.parts) - 1]


def _sha256(path: Path) -> str:
    """Hash one model file without retaining its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
