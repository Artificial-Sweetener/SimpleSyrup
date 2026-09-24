# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discover and resolve Ultralytics checkpoints in ComfyUI model folders."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import TypeAlias, cast

from .model_folders import SUPPORTED_MODEL_EXTENSIONS

ULTRALYTICS_FOLDER = "ultralytics"
ULTRALYTICS_BBOX_FOLDER = "ultralytics_bbox"
ULTRALYTICS_SEGM_FOLDER = "ultralytics_segm"

ModelFolderRegistry: TypeAlias = dict[str, tuple[list[str], set[str]]]


class UltralyticsModelFolders:
    """Own ComfyUI folder registration, discovery, and safe path resolution."""

    def __init__(self, folder_paths_module: ModuleType | None = None) -> None:
        """Create the adapter with an optional ComfyUI module override."""

        self._folder_paths_module = folder_paths_module

    @property
    def folder_paths_module(self) -> ModuleType | None:
        """Return the resolved or injected folder-paths module when available."""

        return self._folder_paths_module

    def available_models(self) -> list[str]:
        """Return supported model files in registered Ultralytics folders."""

        self.register()
        folder_paths = self._folder_paths()
        choices: set[str] = set()

        for folder in self.paths_for(ULTRALYTICS_FOLDER):
            if not folder.is_dir():
                continue
            choices.update(path.name for path in _supported_files(folder))
            choices.update(
                f"bbox/{path.name}" for path in _supported_files(folder / "bbox")
            )
            choices.update(
                f"segm/{path.name}" for path in _supported_files(folder / "segm")
            )

        for path in self.paths_for(ULTRALYTICS_BBOX_FOLDER):
            choices.update(f"bbox/{file.name}" for file in _supported_files(path))
        for path in self.paths_for(ULTRALYTICS_SEGM_FOLDER):
            choices.update(f"segm/{file.name}" for file in _supported_files(path))

        registry = cast(
            ModelFolderRegistry,
            getattr(folder_paths, "folder_names_and_paths", {}),
        )
        for folder_name in (
            ULTRALYTICS_FOLDER,
            ULTRALYTICS_BBOX_FOLDER,
            ULTRALYTICS_SEGM_FOLDER,
        ):
            if folder_name not in registry:
                continue
            for filename in folder_paths.get_filename_list(folder_name):
                path = Path(str(filename))
                if path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
                    continue
                if folder_name == ULTRALYTICS_BBOX_FOLDER:
                    choices.add(f"bbox/{path.name}")
                elif folder_name == ULTRALYTICS_SEGM_FOLDER:
                    choices.add(f"segm/{path.name}")
                else:
                    choices.add(path.as_posix())
        return sorted(choices)

    def resolve(self, model_name: str) -> Path:
        """Resolve a safe model choice inside the configured model folders."""

        safe_name = Path(model_name.replace("\\", "/"))
        if safe_name.is_absolute() or ".." in safe_name.parts:
            raise ValueError(
                f"Ultralytics model name '{model_name}' is not a safe relative path."
            )
        for candidate in self._candidate_paths(safe_name):
            if candidate.is_file():
                return candidate
        raise ValueError(
            f"Ultralytics model '{model_name}' was not found in configured "
            "ComfyUI model folders."
        )

    def register(self) -> None:
        """Register conventional Ultralytics folders with ComfyUI when possible."""

        folder_paths = self._folder_paths()
        models_dir = Path(str(folder_paths.models_dir))
        add_model_folder_path = getattr(folder_paths, "add_model_folder_path", None)
        if add_model_folder_path is None:
            return
        registry = cast(
            ModelFolderRegistry,
            getattr(folder_paths, "folder_names_and_paths", {}),
        )
        registrations = (
            (ULTRALYTICS_FOLDER, models_dir / "ultralytics"),
            (ULTRALYTICS_BBOX_FOLDER, models_dir / "ultralytics" / "bbox"),
            (ULTRALYTICS_SEGM_FOLDER, models_dir / "ultralytics" / "segm"),
        )
        for folder_name, path in registrations:
            if folder_name not in registry:
                add_model_folder_path(folder_name, str(path))

    def paths_for(self, folder_name: str) -> list[Path]:
        """Return registered paths for one ComfyUI model folder."""

        folder_paths = self._folder_paths()
        models_dir = Path(str(folder_paths.models_dir))
        fallback = {
            ULTRALYTICS_FOLDER: models_dir / "ultralytics",
            ULTRALYTICS_BBOX_FOLDER: models_dir / "ultralytics" / "bbox",
            ULTRALYTICS_SEGM_FOLDER: models_dir / "ultralytics" / "segm",
        }[folder_name]
        registry = cast(
            ModelFolderRegistry,
            getattr(folder_paths, "folder_names_and_paths", {}),
        )
        paths = [fallback]
        if folder_name in registry:
            paths = [Path(str(path)) for path in registry[folder_name][0]] + paths
        return _unique_paths(paths)

    def _candidate_paths(self, model_name: Path) -> list[Path]:
        """Return bounded filesystem candidates for a model choice."""

        self.register()
        parts = model_name.parts
        if len(parts) >= 2 and parts[0] == "bbox":
            relative = Path(*parts[1:])
            return [
                *(
                    folder / relative
                    for folder in self.paths_for(ULTRALYTICS_BBOX_FOLDER)
                ),
                *(
                    folder / "bbox" / relative
                    for folder in self.paths_for(ULTRALYTICS_FOLDER)
                ),
            ]
        if len(parts) >= 2 and parts[0] == "segm":
            relative = Path(*parts[1:])
            return [
                *(
                    folder / relative
                    for folder in self.paths_for(ULTRALYTICS_SEGM_FOLDER)
                ),
                *(
                    folder / "segm" / relative
                    for folder in self.paths_for(ULTRALYTICS_FOLDER)
                ),
            ]
        return [folder / model_name for folder in self.paths_for(ULTRALYTICS_FOLDER)]

    def _folder_paths(self) -> ModuleType:
        """Import ComfyUI folder path helpers lazily."""

        if self._folder_paths_module is None:
            module = importlib.import_module("folder_paths")
            if not isinstance(module, ModuleType):
                raise TypeError("folder_paths import did not return a module.")
            self._folder_paths_module = module
        return self._folder_paths_module


def _supported_files(folder: Path) -> list[Path]:
    """Return directly contained supported model files for a folder."""

    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_EXTENSIONS
    )


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Return unique paths while preserving order."""

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique
