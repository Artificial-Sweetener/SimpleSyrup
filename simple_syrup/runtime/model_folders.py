# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI model folder registration and bounded discovery helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
SUPPORTED_MODEL_EXTENSIONS = frozenset({".pt", ".pth", ".safetensors"})
WD14_TAGGER_MODEL_EXTENSIONS = frozenset({".onnx", ".csv"})
MODEL_FOLDER_EXTENSIONS = {
    "sams": SUPPORTED_MODEL_EXTENSIONS,
    "grounding-dino": SUPPORTED_MODEL_EXTENSIONS,
    "vitmatte": SUPPORTED_MODEL_EXTENSIONS,
    "wd14_tagger": WD14_TAGGER_MODEL_EXTENSIONS,
}


def register_required_model_folders(
    folder_paths_module: ModuleType | None = None,
) -> None:
    """Register model folders required by grounded SAM if ComfyUI has not."""

    folder_paths = folder_paths_module or _folder_paths()
    models_dir = Path(str(folder_paths.models_dir))
    _register_folder(
        folder_paths,
        "sams",
        models_dir / "sams",
        MODEL_FOLDER_EXTENSIONS["sams"],
    )
    _register_folder(
        folder_paths,
        "grounding-dino",
        models_dir / "grounding-dino",
        MODEL_FOLDER_EXTENSIONS["grounding-dino"],
    )
    _register_folder(
        folder_paths,
        "vitmatte",
        models_dir / "vitmatte",
        MODEL_FOLDER_EXTENSIONS["vitmatte"],
    )
    _register_folder(
        folder_paths,
        "wd14_tagger",
        models_dir / "wd14_tagger",
        MODEL_FOLDER_EXTENSIONS["wd14_tagger"],
    )


def get_model_folder_paths(
    folder_name: str,
    folder_paths_module: ModuleType | None = None,
) -> list[Path]:
    """Return registered paths plus the conventional fallback for a model folder."""

    folder_paths = folder_paths_module or _folder_paths()
    models_dir = Path(str(folder_paths.models_dir))
    fallback = models_dir / folder_name
    registry = getattr(folder_paths, "folder_names_and_paths", {})
    paths: list[Path] = []

    if folder_name in registry:
        registered_paths = registry[folder_name][0]
        paths.extend(Path(str(path)) for path in registered_paths)

    if fallback not in paths:
        paths.append(fallback)

    return _unique_paths(paths)


def get_primary_model_folder(
    folder_name: str,
    folder_paths_module: ModuleType | None = None,
) -> Path:
    """Return the first registered or fallback path for a model folder."""

    return get_model_folder_paths(folder_name, folder_paths_module)[0]


def resolve_model_file(
    folder_name: str,
    filename: str,
    folder_paths_module: ModuleType | None = None,
) -> Path | None:
    """Return a model file path when it exists in registered or fallback folders."""

    safe_filename = Path(filename)
    if safe_filename.is_absolute() or ".." in safe_filename.parts:
        raise ValueError(f"Model filename '{filename}' is not a safe relative path.")

    for folder in get_model_folder_paths(folder_name, folder_paths_module):
        candidate = folder / safe_filename
        if candidate.is_file():
            return candidate
    return None


def expected_model_file(
    folder_name: str,
    filename: str,
    folder_paths_module: ModuleType | None = None,
) -> Path:
    """Return the expected destination for a known model artifact."""

    safe_filename = Path(filename)
    if safe_filename.is_absolute() or ".." in safe_filename.parts:
        raise ValueError(f"Model filename '{filename}' is not a safe relative path.")
    return get_primary_model_folder(folder_name, folder_paths_module) / safe_filename


def nonrecursive_model_files(
    folder_name: str,
    folder_paths_module: ModuleType | None = None,
) -> list[str]:
    """Return supported files directly inside a model folder without recursion."""

    files: set[str] = set()
    extensions = _folder_extensions(folder_name, folder_paths_module)
    for folder in get_model_folder_paths(folder_name, folder_paths_module):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in extensions:
                files.add(path.name)
    return sorted(files)


def _register_folder(
    folder_paths: ModuleType,
    folder_name: str,
    path: Path,
    extensions: frozenset[str],
) -> None:
    """Register one model folder if ComfyUI has not already done so."""

    registry: dict[str, tuple[list[str], set[str]]] = (
        folder_paths.folder_names_and_paths
    )
    if folder_name in registry:
        return

    add_model_folder_path = folder_paths.add_model_folder_path
    add_model_folder_path(folder_name, str(path))
    registry[folder_name] = (registry[folder_name][0], set(extensions))
    LOGGER.info(
        "registered model folder", extra={"folder_name": folder_name, "path": str(path)}
    )


def _folder_extensions(
    folder_name: str,
    folder_paths_module: ModuleType | None,
) -> frozenset[str]:
    """Return supported extensions for a model folder."""

    folder_paths = folder_paths_module or _folder_paths()
    registry = getattr(folder_paths, "folder_names_and_paths", {})
    if folder_name in registry:
        _paths, extensions = registry[folder_name]
        return frozenset(extensions)
    return MODEL_FOLDER_EXTENSIONS.get(folder_name, SUPPORTED_MODEL_EXTENSIONS)


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Return paths in order without duplicates."""

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        unique.append(path)
        seen.add(key)
    return unique


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module
