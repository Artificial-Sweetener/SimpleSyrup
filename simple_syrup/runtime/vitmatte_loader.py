# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve, download, and load ViTMatte models."""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from ..shared.logging import get_logger
from .loaded_models import LoadedViTMatteModel
from .model_catalog import ModelEntry, get_vitmatte_entry, vitmatte_choices
from .model_device_manager import TorchModelDeviceManager
from .model_downloads import NullProgressReporter, ProgressReporter
from .model_folders import get_primary_model_folder, register_required_model_folders
from .model_instance_cache import ModelInstanceCache

LOGGER = get_logger(__name__)
VITMATTE_CHOICES = tuple(vitmatte_choices())
VITMATTE_SMALL = "vitmatte-small-composition-1k"
VITMATTE_BASE = "vitmatte-base-composition-1k"


class SnapshotDownloader(Protocol):
    """Download a Hugging Face snapshot into a local directory."""

    def download_snapshot(
        self,
        repo_id: str,
        destination: Path,
        progress: ProgressReporter | None = None,
    ) -> Path:
        """Download a model snapshot and return its local directory."""


@dataclass(frozen=True)
class ViTMatteResolution:
    """Resolved ViTMatte directory and source metadata."""

    path: Path
    source: str
    downloaded: bool


@dataclass(frozen=True)
class ViTMatteModelCacheKey:
    """Identify a loaded ViTMatte runtime for process-level reuse."""

    model_id: str
    model_path: Path


_LOADED_VITMATTE_MODELS: dict[ViTMatteModelCacheKey, LoadedViTMatteModel] = {}


class HuggingFaceSnapshotDownloader:
    """Download trusted Hugging Face model snapshots."""

    def download_snapshot(
        self,
        repo_id: str,
        destination: Path,
        progress: ProgressReporter | None = None,
    ) -> Path:
        """Download a Hugging Face snapshot into `destination`."""

        reporter = progress or NullProgressReporter()
        destination.mkdir(parents=True, exist_ok=True)
        reporter.start(f"Downloading {repo_id}", None)
        try:
            huggingface_hub = importlib.import_module("huggingface_hub")
            snapshot_download = huggingface_hub.snapshot_download
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(destination),
                ignore_patterns=["*.md", "*.txt", "onnx", ".git"],
            )
            reporter.finish()
            return destination
        except Exception:
            LOGGER.exception(
                "vitmatte snapshot download failed",
                extra={"repo_id": repo_id, "destination": str(destination)},
            )
            raise


class ViTMatteLoaderService:
    """Resolve, optionally download, and load known ViTMatte models."""

    def __init__(
        self,
        downloader: SnapshotDownloader | None = None,
        folder_paths_module: ModuleType | None = None,
        device_manager: TorchModelDeviceManager | None = None,
        cache: (
            MutableMapping[ViTMatteModelCacheKey, LoadedViTMatteModel] | None
        ) = None,
    ) -> None:
        """Create a ViTMatte loader with injectable external boundaries."""

        self._downloader = downloader or HuggingFaceSnapshotDownloader()
        self._folder_paths_module = folder_paths_module
        self._device_manager = device_manager or TorchModelDeviceManager()
        self._cache: ModelInstanceCache[ViTMatteModelCacheKey, LoadedViTMatteModel] = (
            ModelInstanceCache(cache if cache is not None else _LOADED_VITMATTE_MODELS)
        )

    def load_model(
        self,
        vitmatte_model: str,
        auto_download: bool,
        progress: ProgressReporter | None = None,
    ) -> LoadedViTMatteModel:
        """Load a ViTMatte model and processor without moving it to CUDA."""

        register_required_model_folders(self._folder_paths_module)
        entry = get_vitmatte_entry(vitmatte_model)
        resolution = self.resolve_model_directory(entry, auto_download, progress)
        key = ViTMatteModelCacheKey(
            model_id=entry.entry_id,
            model_path=resolution.path.resolve(),
        )
        already_loaded = key in self._cache.entries
        loaded = self._cache.get_or_load(
            key,
            lambda: self._load_uncached_model(entry, resolution),
        )
        if already_loaded:
            LOGGER.info(
                "ViTMatte model loaded from process cache",
                extra={
                    "operation": "vitmatte_loader",
                    "model": entry.entry_id,
                    "model_path": str(resolution.path),
                    "source": resolution.source,
                },
            )
        return loaded

    def _load_uncached_model(
        self,
        entry: ModelEntry,
        resolution: ViTMatteResolution,
    ) -> LoadedViTMatteModel:
        """Load and wrap ViTMatte after directory resolution and cache lookup."""

        model, processor = self._load_transformers_model(resolution.path)
        managed_model = self._device_manager.manage(
            model,
            model_id=entry.entry_id,
            source=str(resolution.path),
        )
        loaded = LoadedViTMatteModel(
            model=model,
            processor=processor,
            source=resolution.source,
            model_id=entry.entry_id,
            model_path=resolution.path,
            managed_model=managed_model,
        )
        LOGGER.info(
            "ViTMatte model loaded",
            extra={
                "operation": "vitmatte_loader",
                "model": entry.entry_id,
                "model_path": str(resolution.path),
                "source": resolution.source,
            },
        )
        return loaded

    def resolve_model_directory(
        self,
        entry: ModelEntry,
        auto_download: bool,
        progress: ProgressReporter | None = None,
    ) -> ViTMatteResolution:
        """Resolve a valid ViTMatte directory or download to the canonical path."""

        canonical = self._canonical_path(entry)
        if is_valid_vitmatte_directory(canonical):
            return ViTMatteResolution(canonical, "canonical", downloaded=False)

        layerstyle = self._layerstyle_path(entry)
        if layerstyle is not None and is_valid_vitmatte_directory(layerstyle):
            return ViTMatteResolution(layerstyle, "layerstyle-compatible", False)

        if not auto_download:
            layerstyle_hint = f" or compatible LayerStyle path: {layerstyle}"
            raise FileNotFoundError(
                f"ViTMatte model '{entry.display_name}' is missing and "
                f"auto_download is disabled. Expected a valid model at: "
                f"{canonical}{layerstyle_hint}. Enable auto_download on "
                "ViTMatte Model Loader or install the model files."
            )

        downloaded = self._downloader.download_snapshot(
            entry.source_repo,
            canonical,
            progress,
        )
        if not is_valid_vitmatte_directory(downloaded):
            raise FileNotFoundError(
                f"Downloaded ViTMatte files in '{downloaded}' are incomplete."
            )
        return ViTMatteResolution(downloaded, f"downloaded: {entry.source_repo}", True)

    def _canonical_path(self, entry: ModelEntry) -> Path:
        """Return SimpleSyrup's canonical ViTMatte model directory."""

        return (
            get_primary_model_folder("vitmatte", self._folder_paths_module)
            / entry.entry_id
        )

    def _layerstyle_path(self, entry: ModelEntry) -> Path | None:
        """Return LayerStyle's compatible directory for a ViTMatte entry."""

        models_dir = self._models_dir()
        if entry.entry_id == VITMATTE_SMALL:
            return models_dir / "vitmatte"
        if entry.entry_id == VITMATTE_BASE:
            return models_dir / "vitmatte-base-composition-1k"
        return None

    def _models_dir(self) -> Path:
        """Return ComfyUI's models directory."""

        folder_paths = self._folder_paths_module or importlib.import_module(
            "folder_paths"
        )
        return Path(str(folder_paths.models_dir))

    def _load_transformers_model(self, path: Path) -> tuple[object, object]:
        """Load ViTMatte model and processor from a local directory."""

        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError(
                "transformers with ViTMatte support is required to load ViTMatte."
            ) from error

        model_class: Any = transformers.VitMatteForImageMatting
        processor_class: Any = transformers.VitMatteImageProcessor
        model = model_class.from_pretrained(str(path), local_files_only=True)
        processor = processor_class.from_pretrained(str(path), local_files_only=True)
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        return model, processor


def is_valid_vitmatte_directory(path: Path) -> bool:
    """Return whether `path` contains a usable ViTMatte snapshot."""

    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "preprocessor_config.json").is_file()
        and (
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
        )
    )
