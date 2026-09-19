# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ultralytics detector model discovery and lazy loading."""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias, cast

from ..shared.logging import get_logger
from .model_catalog import ULTRALYTICS_ENTRIES, ModelEntry
from .model_choices import ModelChoiceService
from .model_downloads import DownloadRequest, ModelDownloader, ProgressReporter
from .model_folders import (
    SUPPORTED_MODEL_EXTENSIONS,
    expected_model_file,
    resolve_model_file,
)
from .model_instance_cache import ModelInstanceCache

LOGGER = get_logger(__name__)

NO_LOCAL_ULTRALYTICS_MODELS = "No local Ultralytics models found"
ULTRALYTICS_FOLDER = "ultralytics"
ULTRALYTICS_BBOX_FOLDER = "ultralytics_bbox"
ULTRALYTICS_SEGM_FOLDER = "ultralytics_segm"

ModelFolderRegistry: TypeAlias = dict[str, tuple[list[str], set[str]]]


@dataclass(frozen=True)
class UltralyticsDetectorModel:
    """Store a loaded Ultralytics detector with SimpleSyrup metadata."""

    model_name: str
    model_path: Path
    model: Any
    task: str
    names: dict[int, str]
    supports_segmentation: bool


@dataclass(frozen=True)
class LoadedUltralyticsDetector:
    """Bundle native and compatibility detector outputs from the loader."""

    detector_model: UltralyticsDetectorModel
    bbox_detector: object
    segm_detector: object


@dataclass(frozen=True)
class UltralyticsModelCacheKey:
    """Identify a loaded Ultralytics detector for process-level reuse."""

    model_path: Path


_LOADED_ULTRALYTICS_MODELS: dict[
    UltralyticsModelCacheKey, LoadedUltralyticsDetector
] = {}


class UltralyticsLoaderService:
    """Discover and load Ultralytics detector models from ComfyUI folders."""

    def __init__(
        self,
        folder_paths_module: ModuleType | None = None,
        ultralytics_module: ModuleType | None = None,
        downloader: ModelDownloader | None = None,
        choice_service: ModelChoiceService | None = None,
        cache: (
            MutableMapping[UltralyticsModelCacheKey, LoadedUltralyticsDetector] | None
        ) = None,
    ) -> None:
        """Create the loader with injectable runtime modules for tests."""

        self._folder_paths_module = folder_paths_module
        self._ultralytics_module = ultralytics_module
        self._downloader = downloader or ModelDownloader()
        self._choice_service = choice_service or ModelChoiceService(
            folder_paths_module=folder_paths_module
        )
        self._cache: ModelInstanceCache[
            UltralyticsModelCacheKey, LoadedUltralyticsDetector
        ] = ModelInstanceCache(
            cache if cache is not None else _LOADED_ULTRALYTICS_MODELS
        )

    def model_choices(self) -> list[str]:
        """Return curated and local Ultralytics model choices for ComfyUI dropdowns."""

        self._register_model_folders()
        curated_choices = self._choice_service.ultralytics_choices()
        curated_local_paths = {
            _catalog_selection(entry) for entry in ULTRALYTICS_ENTRIES
        }
        choices = curated_choices + [
            choice
            for choice in self.available_models()
            if choice not in curated_local_paths
        ]
        return choices or [NO_LOCAL_ULTRALYTICS_MODELS]

    def available_models(self) -> list[str]:
        """Return supported model files in registered Ultralytics folders."""

        self._register_model_folders()
        folder_paths = self._folder_paths()
        choices: set[str] = set()

        for folder in self._folder_paths_for(ULTRALYTICS_FOLDER):
            if not folder.is_dir():
                continue
            choices.update(path.name for path in _supported_files(folder))
            bbox_dir = folder / "bbox"
            segm_dir = folder / "segm"
            choices.update(f"bbox/{path.name}" for path in _supported_files(bbox_dir))
            choices.update(f"segm/{path.name}" for path in _supported_files(segm_dir))

        for path in self._folder_paths_for(ULTRALYTICS_BBOX_FOLDER):
            choices.update(f"bbox/{file.name}" for file in _supported_files(path))
        for path in self._folder_paths_for(ULTRALYTICS_SEGM_FOLDER):
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

    def load(
        self,
        model_name: str,
        progress: ProgressReporter | None = None,
    ) -> LoadedUltralyticsDetector:
        """Load one Ultralytics model and create compatibility facades."""

        self.reject_sentinel(model_name)
        entry = _catalog_entry_or_none(model_name)
        if entry is None:
            model_path = self.resolve_model_path(model_name)
            normalized_name = _normalized_model_name(model_name)
        else:
            model_path = self._resolve_catalog_entry(entry, progress)
            normalized_name = _catalog_selection(entry)

        key = UltralyticsModelCacheKey(model_path=model_path.resolve())
        already_loaded = key in self._cache.entries
        loaded = self._cache.get_or_load(
            key,
            lambda: self._load_uncached_detector(normalized_name, model_path),
        )
        if already_loaded:
            LOGGER.info(
                "Ultralytics model loaded from process cache",
                extra={
                    "operation": "load_ultralytics_model",
                    "model_name": normalized_name,
                    "model_path": str(model_path),
                    "task": loaded.detector_model.task,
                },
            )
        return loaded

    def _resolve_catalog_entry(
        self,
        entry: ModelEntry,
        progress: ProgressReporter | None,
    ) -> Path:
        """Resolve or securely download one curated Ultralytics checkpoint."""

        if len(entry.artifacts) != 1:
            raise RuntimeError(
                f"Ultralytics catalog entry '{entry.entry_id}' must have one artifact."
            )

        self._register_model_folders()
        artifact = entry.artifacts[0]
        existing = resolve_model_file(
            artifact.folder_name,
            artifact.filename,
            self._folder_paths_module,
        )
        destination = existing or expected_model_file(
            artifact.folder_name, artifact.filename, self._folder_paths_module
        )
        result = self._downloader.download(
            DownloadRequest(
                source_url=artifact.source_url,
                destination_path=destination,
                expected_folder=destination.parent,
                description=artifact.description,
                expected_sha256=artifact.sha256,
            ),
            progress,
        )
        return result.path

    def _load_uncached_detector(
        self,
        model_name: str,
        model_path: Path,
    ) -> LoadedUltralyticsDetector:
        """Load an Ultralytics detector after path resolution and cache lookup."""

        ultralytics_module = self._ultralytics()
        model_class = getattr(ultralytics_module, "YOLO", None)
        if model_class is None:
            raise RuntimeError(
                "Ultralytics support requires a module exposing the YOLO class."
            )

        try:
            raw_model = model_class(str(model_path))
        except Exception as exc:
            LOGGER.error(
                "Failed to load Ultralytics model",
                extra={
                    "operation": "load_ultralytics_model",
                    "model_name": model_name,
                    "model_path": str(model_path),
                },
                exc_info=True,
            )
            raise RuntimeError(
                f"Ultralytics model '{model_name}' could not be loaded from "
                f"'{model_path}'."
            ) from exc

        task = _model_task(model_name, raw_model)
        device_hint = _model_device_hint(raw_model)
        detector_model = UltralyticsDetectorModel(
            model_name=model_name,
            model_path=model_path,
            model=raw_model,
            task=task,
            names=_model_names(raw_model),
            supports_segmentation=task in {"segment", "segm"},
        )

        from .detector_compat import BBoxDetectorFacade, SegmDetectorFacade

        bbox_detector = BBoxDetectorFacade(detector_model)
        segm_detector = SegmDetectorFacade(detector_model, bbox_detector)
        LOGGER.info(
            "Ultralytics model loaded",
            extra={
                "operation": "load_ultralytics_model",
                "model_name": model_name,
                "model_path": str(model_path),
                "task": task,
                "device": device_hint,
            },
        )
        return LoadedUltralyticsDetector(
            detector_model=detector_model,
            bbox_detector=bbox_detector,
            segm_detector=segm_detector,
        )

    def reject_sentinel(self, model_name: str) -> None:
        """Reject placeholder dropdown selections before filesystem work."""

        if model_name == NO_LOCAL_ULTRALYTICS_MODELS:
            raise ValueError(
                "No local Ultralytics models are available. Enable 'Show "
                "downloadable models in loader dropdowns' in SimpleSyrup settings "
                "or install a model in models\\ultralytics, "
                "models\\ultralytics\\bbox, or models\\ultralytics\\segm."
            )

    def resolve_model_path(self, model_name: str) -> Path:
        """Resolve a safe model choice to a file inside ComfyUI model folders."""

        safe_name = Path(model_name.replace("\\", "/"))
        if safe_name.is_absolute() or ".." in safe_name.parts:
            raise ValueError(
                f"Ultralytics model name '{model_name}' is not a safe relative path."
            )

        candidates = self._candidate_paths(safe_name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        raise ValueError(
            f"Ultralytics model '{model_name}' was not found in configured "
            "ComfyUI model folders."
        )

    def _candidate_paths(self, model_name: Path) -> list[Path]:
        """Return bounded filesystem candidates for a model choice."""

        self._register_model_folders()
        candidates: list[Path] = []
        parts = model_name.parts
        if len(parts) >= 2 and parts[0] == "bbox":
            relative = Path(*parts[1:])
            candidates.extend(
                folder / relative
                for folder in self._folder_paths_for(ULTRALYTICS_BBOX_FOLDER)
            )
            candidates.extend(
                folder / "bbox" / relative
                for folder in self._folder_paths_for(ULTRALYTICS_FOLDER)
            )
        elif len(parts) >= 2 and parts[0] == "segm":
            relative = Path(*parts[1:])
            candidates.extend(
                folder / relative
                for folder in self._folder_paths_for(ULTRALYTICS_SEGM_FOLDER)
            )
            candidates.extend(
                folder / "segm" / relative
                for folder in self._folder_paths_for(ULTRALYTICS_FOLDER)
            )
        else:
            candidates.extend(
                folder / model_name
                for folder in self._folder_paths_for(ULTRALYTICS_FOLDER)
            )
        return candidates

    def _register_model_folders(self) -> None:
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
            if folder_name in registry:
                continue
            add_model_folder_path(folder_name, str(path))

    def _folder_paths_for(self, folder_name: str) -> list[Path]:
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

    def _folder_paths(self) -> ModuleType:
        """Import ComfyUI folder path helpers lazily."""

        if self._folder_paths_module is not None:
            return self._folder_paths_module
        module = importlib.import_module("folder_paths")
        if not isinstance(module, ModuleType):
            raise TypeError("folder_paths import did not return a module.")
        self._folder_paths_module = module
        return module

    def _ultralytics(self) -> ModuleType:
        """Import Ultralytics lazily and fail with an actionable message."""

        if self._ultralytics_module is not None:
            return self._ultralytics_module
        try:
            module = importlib.import_module("ultralytics")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ultralytics support requires the 'ultralytics' package in the "
                "ComfyUI virtual environment."
            ) from exc
        if not isinstance(module, ModuleType):
            raise TypeError("ultralytics import did not return a module.")
        self._ultralytics_module = module
        return module


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
        if key in seen:
            continue
        unique.append(path)
        seen.add(key)
    return unique


def _normalized_model_name(model_name: str) -> str:
    """Return a stable model selection string for cache identity."""

    return model_name.replace("\\", "/")


def _catalog_entry_or_none(selection: str) -> ModelEntry | None:
    """Return a curated Ultralytics entry when a dropdown label matches it."""

    return next(
        (
            entry
            for entry in ULTRALYTICS_ENTRIES
            if selection in (entry.entry_id, entry.display_name)
        ),
        None,
    )


def _catalog_selection(entry: ModelEntry) -> str:
    """Return the local conventional selection path for one catalog entry."""

    if len(entry.artifacts) != 1:
        raise ValueError(
            f"Ultralytics catalog entry '{entry.entry_id}' must have one artifact."
        )

    artifact = entry.artifacts[0]
    prefix_by_folder = {
        ULTRALYTICS_BBOX_FOLDER: "bbox",
        ULTRALYTICS_SEGM_FOLDER: "segm",
    }
    try:
        prefix = prefix_by_folder[artifact.folder_name]
    except KeyError as error:
        raise ValueError(
            f"Ultralytics catalog entry '{entry.entry_id}' has unsupported folder "
            f"'{artifact.folder_name}'."
        ) from error
    return f"{prefix}/{artifact.filename}"


def _model_task(model_name: str, raw_model: object) -> str:
    """Infer detector task from choice prefix or model metadata."""

    normalized_name = model_name.replace("\\", "/")
    if normalized_name.startswith("segm/"):
        return "segment"
    if normalized_name.startswith("bbox/"):
        return "detect"

    task = getattr(raw_model, "task", None)
    if isinstance(task, str) and task:
        return task
    return "detect"


def _model_names(raw_model: object) -> dict[int, str]:
    """Extract class names from a loaded Ultralytics model."""

    names = getattr(raw_model, "names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _model_device_hint(raw_model: object) -> str:
    """Return a best-effort Ultralytics device hint for diagnostics."""

    direct_device = getattr(raw_model, "device", None)
    if direct_device is not None:
        return str(direct_device)
    inner_model = getattr(raw_model, "model", None)
    inner_device = getattr(inner_model, "device", None)
    if inner_device is not None:
        return str(inner_device)
    return "runtime-owned"
