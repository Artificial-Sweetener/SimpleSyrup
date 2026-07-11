# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load SAM models for ComfyUI model-loader nodes."""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ..shared.logging import get_logger
from .loaded_models import LoadedSAMModel
from .model_catalog import ModelEntry, get_sam_entry
from .model_device_manager import TorchModelDeviceManager
from .model_downloads import DownloadRequest, ModelDownloader, ProgressReporter
from .model_folders import (
    expected_model_file,
    register_required_model_folders,
    resolve_model_file,
)
from .model_instance_cache import ModelInstanceCache
from .progress import NullPhaseProgressReporter, PhaseProgressReporter

LOGGER = get_logger(__name__)
SAM_HQ_RUNTIME_PACKAGE = "simple_syrup.third_party.sam_hq_runtime"


@dataclass(frozen=True)
class SAMModelCacheKey:
    """Identify a loaded SAM model for process-level reuse."""

    model_id: str
    model_type: str
    checkpoint_path: Path


_LOADED_SAM_MODELS: dict[SAMModelCacheKey, LoadedSAMModel] = {}


class SAMLoaderService:
    """Resolve, download, and load known SAM models."""

    def __init__(
        self,
        downloader: ModelDownloader | None = None,
        folder_paths_module: ModuleType | None = None,
        device_manager: TorchModelDeviceManager | None = None,
        cache: MutableMapping[SAMModelCacheKey, LoadedSAMModel] | None = None,
    ) -> None:
        """Create a SAM loader with injectable external boundaries."""

        self._downloader = downloader or ModelDownloader()
        self._folder_paths_module = folder_paths_module
        self._device_manager = device_manager or TorchModelDeviceManager()
        self._cache: ModelInstanceCache[SAMModelCacheKey, LoadedSAMModel] = (
            ModelInstanceCache(cache if cache is not None else _LOADED_SAM_MODELS)
        )

    def load_model(
        self,
        sam_model: str,
        auto_download: bool,
        progress: ProgressReporter | None = None,
        phase_progress: PhaseProgressReporter | None = None,
    ) -> LoadedSAMModel:
        """Load a known SAM model and return a `SAM_MODEL`-compatible object."""

        reporter = phase_progress or NullPhaseProgressReporter()
        reporter.advance("resolving_artifacts")
        try:
            register_required_model_folders(self._folder_paths_module)
            entry = get_sam_entry(sam_model)
            artifact_paths = self._resolve_artifacts(entry, auto_download, progress)
            checkpoint_path = artifact_paths[0]
            key = SAMModelCacheKey(
                model_id=entry.entry_id,
                model_type=entry.model_type,
                checkpoint_path=checkpoint_path.resolve(),
            )
            already_loaded = key in self._cache.entries
            if already_loaded:
                reporter.advance("cache_hit")
            loaded = self._cache.get_or_load(
                key,
                lambda: self._load_uncached_model(entry, checkpoint_path, reporter),
            )
            if already_loaded:
                LOGGER.info(
                    "SAM model loaded from process cache",
                    extra={
                        "operation": "sam_loader",
                        "model": entry.entry_id,
                        "model_type": entry.model_type,
                        "checkpoint_path": str(checkpoint_path),
                    },
                )
            reporter.advance("completed")
            return loaded
        except Exception:
            reporter.advance("failed")
            LOGGER.exception(
                "SAM model load failed",
                extra={"operation": "sam_loader", "model": sam_model},
            )
            raise

    def _load_uncached_model(
        self,
        entry: ModelEntry,
        checkpoint_path: Path,
        phase_progress: PhaseProgressReporter,
    ) -> LoadedSAMModel:
        """Load and wrap a SAM model after artifact resolution and cache lookup."""

        phase_progress.advance("loading_checkpoint")
        model = self._load_segment_anything_model(entry, checkpoint_path)
        phase_progress.advance("registering_device_management")
        managed_model = self._device_manager.manage(
            model,
            model_id=entry.entry_id,
            source=str(checkpoint_path),
        )
        LOGGER.info(
            "SAM model loaded",
            extra={
                "operation": "sam_loader",
                "model": entry.entry_id,
                "model_type": entry.model_type,
                "checkpoint_path": str(checkpoint_path),
            },
        )
        return LoadedSAMModel(
            model=model,
            source=str(checkpoint_path),
            model_id=entry.entry_id,
            managed_model=managed_model,
        )

    def _resolve_artifacts(
        self,
        entry: ModelEntry,
        auto_download: bool,
        progress: ProgressReporter | None,
    ) -> list[Path]:
        """Resolve or download every SAM artifact for a catalog entry."""

        paths: list[Path] = []
        for artifact in entry.artifacts:
            existing = resolve_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            if existing is not None:
                paths.append(existing)
                continue

            destination = expected_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            if not auto_download or not entry.auto_download_allowed:
                raise FileNotFoundError(
                    f"SAM model '{entry.display_name}' is missing and "
                    f"auto_download is disabled. Expected: {destination}. "
                    "Enable auto_download on SAM Model Loader or install the "
                    "model file."
                )
            result = self._downloader.download(
                DownloadRequest(
                    source_url=artifact.source_url,
                    destination_path=destination,
                    expected_folder=destination.parent,
                    description=artifact.description,
                ),
                progress,
            )
            paths.append(result.path)
        return paths

    def _load_segment_anything_model(
        self,
        entry: ModelEntry,
        checkpoint_path: Path,
    ) -> object:
        """Load a SAM model from the segment-anything registry."""

        try:
            importlib.invalidate_caches()
            runtime_module = importlib.import_module(
                _registry_module_name(entry.model_type)
            )
        except ImportError as error:
            raise RuntimeError(
                _registry_import_error_message(entry.model_type, error)
            ) from error

        registry: dict[str, Any] = runtime_module.sam_model_registry
        model = registry[entry.model_type](checkpoint=str(checkpoint_path))
        model.model_name = checkpoint_path.name
        return model


def _registry_module_name(model_type: str) -> str:
    """Return the registry module that owns one SAM-compatible model type."""

    if model_type.startswith("sam_hq") or model_type == "mobile_sam":
        return f"{SAM_HQ_RUNTIME_PACKAGE}.build_sam_hq"
    return "segment_anything"


def _registry_import_error_message(model_type: str, error: ImportError) -> str:
    """Return an actionable registry import failure for a SAM model type."""

    if model_type.startswith("sam_hq"):
        return (
            "SAM-HQ support requires SimpleSyrup's bundled SAM-HQ runtime and "
            "its dependencies. Reinstall SimpleSyrup or restore "
            f"{SAM_HQ_RUNTIME_PACKAGE}. Import failed: {error}."
        )
    if model_type == "mobile_sam":
        return (
            "MobileSAM support requires SimpleSyrup's bundled SAM-HQ/MobileSAM "
            "runtime and its dependencies. Reinstall SimpleSyrup or restore "
            f"{SAM_HQ_RUNTIME_PACKAGE}. Import failed: {error}."
        )
    return f"segment-anything is required to load SAM models. Import failed: {error}."
