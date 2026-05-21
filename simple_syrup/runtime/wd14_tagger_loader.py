# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve, download, and load WD14 tagger models."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from ..shared.logging import get_logger
from .loaded_models import LoadedWD14Tagger
from .model_catalog import ModelEntry, get_wd14_tagger_entry
from .model_downloads import (
    DownloadRequest,
    DownloadResult,
    ModelDownloader,
    ProgressReporter,
)
from .model_folders import (
    expected_model_file,
    register_required_model_folders,
    resolve_model_file,
)
from .model_instance_cache import ModelInstanceCache
from .wd14_tagger import WD14Session, WD14TagRecord, load_wd14_tags

LOGGER = get_logger(__name__)
DEFAULT_WD14_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")

SessionFactory = Callable[[Path, tuple[str, ...]], WD14Session]
TagLoader = Callable[[Path], tuple[WD14TagRecord, ...]]


class ArtifactDownloader(Protocol):
    """Download one trusted model artifact."""

    def download(
        self,
        request: DownloadRequest,
        progress: ProgressReporter | None = None,
    ) -> DownloadResult:
        """Download an artifact and return its resolved path."""


@dataclass(frozen=True)
class WD14TaggerCacheKey:
    """Identify a loaded WD14 runtime for process-level reuse."""

    model_id: str
    onnx_path: Path
    csv_path: Path
    providers: tuple[str, ...]


@dataclass(frozen=True)
class WD14TaggerArtifactPaths:
    """Store resolved WD14 model artifact paths and source metadata."""

    onnx_path: Path
    csv_path: Path
    source: str
    downloaded: bool


_LOADED_WD14_TAGGERS: dict[WD14TaggerCacheKey, LoadedWD14Tagger] = {}


class WD14TaggerLoaderService:
    """Resolve, download, and load known WD14 tagger models."""

    def __init__(
        self,
        downloader: ArtifactDownloader | None = None,
        folder_paths_module: ModuleType | None = None,
        session_factory: SessionFactory | None = None,
        tag_loader: TagLoader | None = None,
        providers: tuple[str, ...] | None = None,
        cache: MutableMapping[WD14TaggerCacheKey, LoadedWD14Tagger] | None = None,
    ) -> None:
        """Create a WD14 loader with injectable external boundaries."""

        self._downloader = downloader or ModelDownloader()
        self._folder_paths_module = folder_paths_module
        self._session_factory = session_factory or _create_onnx_session
        self._tag_loader = tag_loader or load_wd14_tags
        self._providers = providers
        self._cache: ModelInstanceCache[WD14TaggerCacheKey, LoadedWD14Tagger] = (
            ModelInstanceCache(cache if cache is not None else _LOADED_WD14_TAGGERS)
        )

    def load_model(
        self,
        wd14_model: str,
        auto_download: bool,
        progress: ProgressReporter | None = None,
    ) -> LoadedWD14Tagger:
        """Load a known WD14 tagger and return a `WD14_TAGGER` object."""

        register_required_model_folders(self._folder_paths_module)
        entry = get_wd14_tagger_entry(wd14_model)
        artifacts = self._resolve_artifacts(entry, auto_download, progress)
        providers = self._available_providers()
        key = WD14TaggerCacheKey(
            model_id=entry.entry_id,
            onnx_path=artifacts.onnx_path.resolve(),
            csv_path=artifacts.csv_path.resolve(),
            providers=providers,
        )
        already_loaded = key in self._cache.entries
        loaded = self._cache.get_or_load(
            key,
            lambda: self._load_uncached_tagger(entry, artifacts, providers),
        )
        if already_loaded:
            LOGGER.info(
                "WD14 tagger loaded from process cache",
                extra={
                    "operation": "wd14_tagger_loader",
                    "model": entry.entry_id,
                    "onnx_path": str(artifacts.onnx_path),
                    "csv_path": str(artifacts.csv_path),
                    "providers": providers,
                },
            )
        return loaded

    def _load_uncached_tagger(
        self,
        entry: ModelEntry,
        artifacts: WD14TaggerArtifactPaths,
        providers: tuple[str, ...],
    ) -> LoadedWD14Tagger:
        """Load a WD14 tagger after artifact resolution and cache lookup."""

        session = self._session_factory(artifacts.onnx_path, providers)
        tags = self._tag_loader(artifacts.csv_path)
        loaded = LoadedWD14Tagger(
            model_id=entry.entry_id,
            source=artifacts.source,
            onnx_path=artifacts.onnx_path,
            csv_path=artifacts.csv_path,
            providers=providers,
            session=session,
            tags=tags,
        )
        LOGGER.info(
            "WD14 tagger loaded",
            extra={
                "operation": "wd14_tagger_loader",
                "model": entry.entry_id,
                "onnx_path": str(artifacts.onnx_path),
                "csv_path": str(artifacts.csv_path),
                "providers": providers,
                "downloaded": artifacts.downloaded,
            },
        )
        return loaded

    def _resolve_artifacts(
        self,
        entry: ModelEntry,
        auto_download: bool,
        progress: ProgressReporter | None,
    ) -> WD14TaggerArtifactPaths:
        """Resolve or download the ONNX and tag CSV artifacts for an entry."""

        paths: dict[str, Path] = {}
        downloaded = False
        for artifact in entry.artifacts:
            existing = resolve_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            if existing is not None:
                paths[artifact.artifact_id] = existing
                continue

            destination = expected_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            if not auto_download or not entry.auto_download_allowed:
                raise FileNotFoundError(
                    f"WD14 tagger model '{entry.display_name}' is missing and "
                    "auto_download is disabled. Expected: "
                    f"{_expected_paths(entry)}. Enable auto_download on Load "
                    "WD14 Tagger or install the ONNX and CSV files."
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
            paths[artifact.artifact_id] = result.path
            downloaded = downloaded or not result.skipped_existing

        onnx_path = _required_artifact(paths, "onnx", entry)
        csv_path = _required_artifact(paths, "tags", entry)
        source = f"downloaded: {entry.source_repo}" if downloaded else "local"
        return WD14TaggerArtifactPaths(
            onnx_path=onnx_path,
            csv_path=csv_path,
            source=source,
            downloaded=downloaded,
        )

    def _available_providers(self) -> tuple[str, ...]:
        """Return requested ONNX Runtime providers with unavailable entries removed."""

        requested = self._providers or DEFAULT_WD14_PROVIDERS
        ort = import_module("onnxruntime")
        available = set(cast(list[str], ort.get_available_providers()))
        providers = tuple(provider for provider in requested if provider in available)
        if providers:
            if providers != requested:
                LOGGER.warning(
                    "WD14 provider fallback selected",
                    extra={
                        "operation": "wd14_tagger_loader",
                        "requested_providers": requested,
                        "providers": providers,
                    },
                )
            else:
                LOGGER.debug(
                    "WD14 providers selected",
                    extra={
                        "operation": "wd14_tagger_loader",
                        "requested_providers": requested,
                        "providers": providers,
                    },
                )
            return providers
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(
                "No requested ONNX Runtime providers are available for WD14 tagger. "
                f"Requested: {requested}. Available: {tuple(sorted(available))}."
            )
        LOGGER.warning(
            "WD14 provider fallback selected",
            extra={
                "operation": "wd14_tagger_loader",
                "requested_providers": requested,
                "providers": ("CPUExecutionProvider",),
            },
        )
        return ("CPUExecutionProvider",)


def _create_onnx_session(onnx_path: Path, providers: tuple[str, ...]) -> WD14Session:
    """Create an ONNX Runtime inference session."""

    ort = import_module("onnxruntime")
    return cast(WD14Session, ort.InferenceSession(str(onnx_path), providers=providers))


def _required_artifact(
    paths: dict[str, Path],
    artifact_id: str,
    entry: ModelEntry,
) -> Path:
    """Return a required artifact path or fail with catalog context."""

    path = paths.get(artifact_id)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"WD14 tagger model '{entry.display_name}' is incomplete. Expected: "
            f"{_expected_paths(entry)}."
        )
    return path


def _expected_paths(entry: ModelEntry) -> str:
    """Return a readable list of expected artifact filenames for an entry."""

    return " and ".join(artifact.filename for artifact in entry.artifacts)
