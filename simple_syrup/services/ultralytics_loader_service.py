# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Orchestrate Ultralytics discovery, download, loading, and compatibility."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ..runtime.model_catalog import ULTRALYTICS_ENTRIES, ModelEntry
from ..runtime.model_choices import ModelChoiceService
from ..runtime.model_downloads import DownloadRequest, ModelDownloader, ProgressReporter
from ..runtime.model_folders import expected_model_file, resolve_model_file
from ..runtime.model_instance_cache import ModelInstanceCache
from ..runtime.ultralytics_model_adapter import (
    UltralyticsDetectorModel,
    UltralyticsModelAdapter,
)
from ..runtime.ultralytics_model_folders import (
    ULTRALYTICS_BBOX_FOLDER,
    ULTRALYTICS_SEGM_FOLDER,
    UltralyticsModelFolders,
)
from ..shared.logging import get_logger
from .detector_compat import BBoxDetectorFacade, SegmDetectorFacade

LOGGER = get_logger(__name__)

NO_LOCAL_ULTRALYTICS_MODELS = "No local Ultralytics models found"


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
    """Coordinate model selection with runtime adapters and detector facades."""

    def __init__(
        self,
        folder_paths_module: ModuleType | None = None,
        ultralytics_module: ModuleType | None = None,
        downloader: ModelDownloader | None = None,
        choice_service: ModelChoiceService | None = None,
        cache: (
            MutableMapping[UltralyticsModelCacheKey, LoadedUltralyticsDetector] | None
        ) = None,
        model_folders: UltralyticsModelFolders | None = None,
        model_adapter: UltralyticsModelAdapter | None = None,
    ) -> None:
        """Create the service with injectable runtime boundaries."""

        self._model_folders = model_folders or UltralyticsModelFolders(
            folder_paths_module
        )
        self._model_adapter = model_adapter or UltralyticsModelAdapter(
            ultralytics_module
        )
        self._downloader = downloader or ModelDownloader()
        self._choice_service = choice_service or ModelChoiceService()
        self._cache: ModelInstanceCache[
            UltralyticsModelCacheKey, LoadedUltralyticsDetector
        ] = ModelInstanceCache(
            cache if cache is not None else _LOADED_ULTRALYTICS_MODELS
        )

    def model_choices(self) -> list[str]:
        """Return installed choices followed by downloadable catalog choices."""

        self._model_folders.register()
        curated_choices = self._choice_service.ultralytics_choices()
        catalog_choice_labels = {
            _catalog_selection(entry): entry.display_name
            for entry in ULTRALYTICS_ENTRIES
        }
        available_choices = self.available_models()
        visible_catalog_choices = set(curated_choices)
        installed_catalog_choices = [
            entry.display_name
            for entry in ULTRALYTICS_ENTRIES
            if (
                entry.display_name in visible_catalog_choices
                and _catalog_selection(entry) in available_choices
            )
        ]
        installed_non_catalog_choices = [
            choice
            for choice in available_choices
            if choice not in catalog_choice_labels
        ]
        downloadable_choices = [
            choice
            for choice in curated_choices
            if choice not in installed_catalog_choices
        ]
        choices = (
            installed_non_catalog_choices
            + installed_catalog_choices
            + downloadable_choices
        )
        return choices or [NO_LOCAL_ULTRALYTICS_MODELS]

    def available_models(self) -> list[str]:
        """Return model choices discovered by the runtime folder adapter."""

        return self._model_folders.available_models()

    def load(
        self,
        model_name: str,
        progress: ProgressReporter | None = None,
    ) -> LoadedUltralyticsDetector:
        """Load one model and construct its workflow-compatible facades."""

        self.reject_sentinel(model_name)
        entry = _catalog_entry_or_none(model_name)
        if entry is None:
            model_path = self.resolve_model_path(model_name)
            normalized_name = model_name.replace("\\", "/")
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
        """Resolve a safe model selection through the folder adapter."""

        return self._model_folders.resolve(model_name)

    def _resolve_catalog_entry(
        self,
        entry: ModelEntry,
        progress: ProgressReporter | None,
    ) -> Path:
        """Resolve or securely download one curated detector checkpoint."""

        if len(entry.artifacts) != 1:
            raise RuntimeError(
                f"Ultralytics catalog entry '{entry.entry_id}' must have one artifact."
            )
        self._model_folders.register()
        artifact = entry.artifacts[0]
        folder_paths_module = self._model_folders.folder_paths_module
        existing = resolve_model_file(
            artifact.folder_name,
            artifact.filename,
            folder_paths_module,
        )
        destination = existing or expected_model_file(
            artifact.folder_name,
            artifact.filename,
            folder_paths_module,
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
        """Load a native model and construct both compatibility facades."""

        detector_model = self._model_adapter.load(model_name, model_path)
        bbox_detector = BBoxDetectorFacade(detector_model)
        segm_detector = SegmDetectorFacade(detector_model, bbox_detector)
        return LoadedUltralyticsDetector(
            detector_model=detector_model,
            bbox_detector=bbox_detector,
            segm_detector=segm_detector,
        )


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
