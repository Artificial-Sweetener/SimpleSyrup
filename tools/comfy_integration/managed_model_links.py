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


class ManagedComfyModelLinks:
    """Own temporary model visibility beneath known Comfy categories."""

    _CATEGORIES = frozenset(
        {"checkpoints", "diffusion_models", "loras", "text_encoders", "vae"}
    )

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

    def __enter__(self) -> ManagedComfyModelLinks:
        """Create every exact link transactionally after collision validation."""

        if any(target.exists() for target in self._targets):
            raise FileExistsError("An owned Comfy model link already exists.")
        directories = self._required_directories()
        created: list[Path] = []
        linked: list[Path] = []
        try:
            for directory in directories:
                if not directory.exists():
                    directory.mkdir()
                    created.append(directory)
            for link, target in zip(self._links, self._targets, strict=True):
                os.link(link.source.resolve(), target)
                linked.append(target)
        except BaseException:
            self._remove_owned(linked, created)
            raise
        self._created_directories = tuple(created)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove only links and empty directories created by this owner."""

        del exc_type, exc, traceback
        self._remove_owned(list(self._targets), list(self._created_directories))
        self.cleaned = not any(target.exists() for target in self._targets)

    def _validate(self) -> None:
        """Fail closed on roots, sources, categories, and relative targets."""

        if not self._model_root.is_dir():
            raise FileNotFoundError("Comfy model root does not exist.")
        if not self._links:
            raise ValueError("Managed Comfy model links cannot be empty.")
        targets: set[Path] = set()
        for link, target in zip(self._links, self._targets, strict=True):
            if not link.source.resolve().is_file():
                raise FileNotFoundError(
                    f"Required managed model is missing: {link.source}"
                )
            if link.category not in self._CATEGORIES:
                raise ValueError(f"Unsupported Comfy model category: {link.category!r}")
            category_root = self._model_root / link.category
            if not category_root.is_dir():
                raise FileNotFoundError(
                    f"Comfy model category does not exist: {link.category}"
                )
            relative = Path(link.selection_name)
            if (
                relative.is_absolute()
                or len(relative.parts) < 2
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError(
                    "Managed model selections must be safe nested relative paths."
                )
            if target in targets:
                raise ValueError("Managed Comfy model targets must be unique.")
            targets.add(target)

    def _required_directories(self) -> tuple[Path, ...]:
        """Return missing-capable target parents in parent-before-child order."""

        directories: set[Path] = set()
        for link, target in zip(self._links, self._targets, strict=True):
            category_root = self._model_root / link.category
            current = target.parent
            while current != category_root:
                directories.add(current)
                current = current.parent
        return tuple(sorted(directories, key=lambda path: len(path.parts)))

    def _target(self, link: ManagedModelLink) -> Path:
        """Return one target beneath the declared Comfy model category."""

        return self._model_root / link.category / Path(link.selection_name)

    @staticmethod
    def _remove_owned(targets: list[Path], directories: list[Path]) -> None:
        """Remove owned targets and then their empty directories in reverse order."""

        for target in reversed(targets):
            target.unlink(missing_ok=True)
        for directory in sorted(
            directories, key=lambda path: len(path.parts), reverse=True
        ):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
