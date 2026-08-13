# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose exact external model files through cleanup-owned Comfy hard links."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManagedModelLink:
    """Declare one source and one safe Comfy-relative target name."""

    source: Path
    category: str
    selection_name: str


class ManagedSdxlVisualModelLinks:
    """Own one temporary checkpoint link and all visual-matrix LoRA links."""

    _DIRECTORY_NAME = "simple_syrup_u11"

    def __init__(
        self,
        *,
        model_root: Path,
        links: tuple[ManagedModelLink, ...],
    ) -> None:
        """Validate immutable inputs without changing filesystem state."""

        self._model_root = model_root.resolve()
        self._links = links
        self._targets = tuple(self._target(link) for link in links)
        self._created_directories: tuple[Path, ...] = ()
        self.cleaned = False
        self._validate()

    def __enter__(self) -> ManagedSdxlVisualModelLinks:
        """Create all exact links transactionally after collision validation."""

        if any(target.exists() for target in self._targets):
            raise FileExistsError("An owned U11 Comfy model link already exists.")
        directories = tuple(dict.fromkeys(target.parent for target in self._targets))
        created: list[Path] = []
        linked: list[Path] = []
        try:
            for directory in directories:
                if not directory.exists():
                    directory.mkdir(parents=False)
                    created.append(directory)
            for link, target in zip(self._links, self._targets, strict=True):
                os.link(link.source.resolve(), target)
                linked.append(target)
        except BaseException:
            for target in reversed(linked):
                target.unlink(missing_ok=True)
            for directory in reversed(created):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            raise
        self._created_directories = tuple(created)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove only links and empty directories created by this owner."""

        del exc_type, exc, traceback
        for target in reversed(self._targets):
            target.unlink(missing_ok=True)
        for directory in reversed(self._created_directories):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        self.cleaned = not any(target.exists() for target in self._targets)

    def _validate(self) -> None:
        """Fail closed on roots, sources, categories, and target names."""

        if not self._model_root.is_dir():
            raise FileNotFoundError("Comfy model root does not exist.")
        if not self._links:
            raise ValueError("Managed U11 model links cannot be empty.")
        targets: set[Path] = set()
        for link, target in zip(self._links, self._targets, strict=True):
            if not link.source.resolve().is_file():
                raise FileNotFoundError(f"Required U11 model is missing: {link.source}")
            if link.category not in {"checkpoints", "loras"}:
                raise ValueError(f"Unsupported Comfy model category: {link.category!r}")
            relative = Path(link.selection_name)
            if (
                relative.is_absolute()
                or len(relative.parts) != 2
                or relative.parts[0] != self._DIRECTORY_NAME
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError("U11 selection names must use the owned directory.")
            if target in targets:
                raise ValueError("U11 managed model targets must be unique.")
            targets.add(target)

    def _target(self, link: ManagedModelLink) -> Path:
        """Return one target beneath the declared Comfy model category."""

        return self._model_root / link.category / Path(link.selection_name)
