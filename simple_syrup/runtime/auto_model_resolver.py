# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve trusted automatic model artifacts from cache, disk, or download."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType
from typing import Protocol

from ..shared.logging import get_logger
from .auto_model_artifact import AutoModelArtifact
from .auto_model_cache import AutoModelCache, AutoModelCacheEntry, CacheSource
from .model_downloads import (
    DownloadRequest,
    DownloadResult,
    ModelDownloader,
    ProgressReporter,
    sha256_file,
)
from .model_folders import get_model_folder_paths

LOGGER = get_logger(__name__)


class AutoModelDownloadBoundary(Protocol):
    """Downloader interface required by automatic model resolution."""

    def download(
        self,
        request: DownloadRequest,
        progress: ProgressReporter | None = None,
    ) -> DownloadResult:
        """Download one trusted artifact and return its final path."""


@dataclass(frozen=True)
class AutoModelResolution:
    """Resolved automatic model path and provenance."""

    path: Path
    source: CacheSource


class AutoModelResolver:
    """Resolve known model artifacts while maintaining a self-healing cache."""

    def __init__(
        self,
        cache: AutoModelCache | None = None,
        downloader: AutoModelDownloadBoundary | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a resolver with injectable persistence and download boundaries."""

        self._folder_paths_module = folder_paths_module
        self._cache = cache or AutoModelCache(folder_paths_module)
        self._downloader = downloader or ModelDownloader()

    def resolve(
        self,
        artifact: AutoModelArtifact,
        progress: ProgressReporter | None = None,
    ) -> AutoModelResolution:
        """Return a valid local path, downloading and caching when necessary."""

        entries = self._cache.load()
        cached = entries.get(artifact.cache_id)
        if cached is not None and self._cache_entry_is_valid(cached, artifact):
            if cached.file_size is None or cached.modified_time_ns is None:
                self._save_resolution(artifact, cached.path, cached.source)
            LOGGER.info(
                "auto model cache hit",
                extra={"cache_id": artifact.cache_id, "path": str(cached.path)},
            )
            return AutoModelResolution(cached.path, "cached")
        if cached is not None:
            LOGGER.warning(
                "auto model cache entry is stale",
                extra={"cache_id": artifact.cache_id, "path": str(cached.path)},
            )

        found = find_model_artifact(artifact, self._folder_paths_module)
        if found is not None:
            self._save_resolution(artifact, found, "found")
            LOGGER.info(
                "auto model found on disk",
                extra={"cache_id": artifact.cache_id, "path": str(found)},
            )
            return AutoModelResolution(found, "found")

        destination = canonical_auto_destination(artifact, self._folder_paths_module)
        root = _containing_model_root(
            artifact.folder_name,
            destination,
            self._folder_paths_module,
        )
        result = self._downloader.download(
            DownloadRequest(
                source_url=artifact.source_url,
                destination_path=destination,
                expected_folder=root,
                description=artifact.description,
                expected_sha256=artifact.sha256,
            ),
            progress,
        )
        self._save_resolution(artifact, result.path, "downloaded")
        LOGGER.info(
            "auto model downloaded",
            extra={"cache_id": artifact.cache_id, "path": str(result.path)},
        )
        return AutoModelResolution(result.path, "downloaded")

    def _save_resolution(
        self,
        artifact: AutoModelArtifact,
        path: Path,
        source: CacheSource,
    ) -> None:
        """Persist one successful automatic model resolution."""

        file_stat = path.stat()
        self._cache.save_entry(
            artifact.cache_id,
            AutoModelCacheEntry(
                folder_name=artifact.folder_name,
                filename=artifact.filename,
                path=path,
                source=source,
                sha256=artifact.sha256,
                file_size=file_stat.st_size,
                modified_time_ns=file_stat.st_mtime_ns,
            ),
        )

    def _cache_entry_is_valid(
        self,
        entry: AutoModelCacheEntry,
        artifact: AutoModelArtifact,
    ) -> bool:
        """Return whether a remembered entry still resolves to the artifact."""

        if entry.folder_name != artifact.folder_name:
            return False
        if entry.filename != artifact.filename:
            return False
        if entry.sha256 != artifact.sha256:
            return False
        if entry.path.name != artifact.filename:
            return False
        if not entry.path.is_file():
            return False
        try:
            _containing_model_root(
                artifact.folder_name,
                entry.path,
                self._folder_paths_module,
            )
        except ValueError:
            return False
        stat = entry.path.stat()
        if entry.file_size is not None and entry.modified_time_ns is not None:
            return (
                entry.file_size == stat.st_size
                and entry.modified_time_ns == stat.st_mtime_ns
            )
        return sha256_file(entry.path).lower() == artifact.sha256.lower()


def find_model_artifact(
    artifact: AutoModelArtifact,
    folder_paths_module: ModuleType | None = None,
) -> Path | None:
    """Return the first same-named local file matching the catalog checksum."""

    _validate_basename(artifact.filename)
    for root in get_model_folder_paths(artifact.folder_name, folder_paths_module):
        if not root.is_dir():
            continue
        matches = sorted(
            path for path in root.rglob(artifact.filename) if path.is_file()
        )
        for match in matches:
            if match.name != artifact.filename or not _path_is_under(match, root):
                continue
            if sha256_file(match).lower() == artifact.sha256.lower():
                return match
            LOGGER.warning(
                "same-named auto model artifact has a different checksum",
                extra={
                    "cache_id": artifact.cache_id,
                    "path": str(match),
                    "artifact_filename": artifact.filename,
                },
            )
    return None


def find_model_by_basename(
    folder_name: str,
    basename: str,
    folder_paths_module: ModuleType | None = None,
) -> Path | None:
    """Return the first matching model file under registered folder paths."""

    _validate_basename(basename)
    for root in get_model_folder_paths(folder_name, folder_paths_module):
        if not root.is_dir():
            continue
        matches = sorted(path for path in root.rglob(basename) if path.is_file())
        for match in matches:
            if match.name == basename and _path_is_under(match, root):
                return match
    return None


def canonical_auto_destination(
    artifact: AutoModelArtifact,
    folder_paths_module: ModuleType | None = None,
) -> Path:
    """Return the canonical download path under the first registered model folder."""

    _validate_basename(artifact.filename)
    subfolder = _safe_relative_path(artifact.canonical_subfolder)
    root = get_model_folder_paths(artifact.folder_name, folder_paths_module)[0]
    return root / subfolder / artifact.filename


def relative_model_name(
    folder_name: str,
    path: Path,
    folder_paths_module: ModuleType | None = None,
) -> str:
    """Return the ComfyUI-relative filename for a resolved model path."""

    root = _containing_model_root(folder_name, path, folder_paths_module)
    return str(path.resolve().relative_to(root.resolve()))


def _containing_model_root(
    folder_name: str,
    path: Path,
    folder_paths_module: ModuleType | None = None,
) -> Path:
    """Return the registered model root containing a path or raise."""

    resolved_path = path.resolve()
    for root in get_model_folder_paths(folder_name, folder_paths_module):
        resolved_root = root.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            continue
        return root
    raise ValueError(
        f"Model path '{path}' is outside registered '{folder_name}' folders."
    )


def _path_is_under(path: Path, root: Path) -> bool:
    """Return whether a path resolves below a root."""

    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_relative_path(value: str) -> Path:
    """Return a safe relative path from trusted catalog metadata."""

    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Catalog relative path '{value}' is not safe.")
    return path


def _validate_basename(value: str) -> None:
    """Reject unsafe or non-basename model filenames."""

    paths = (Path(value), PurePosixPath(value), PureWindowsPath(value))
    if any(
        path.is_absolute() or path.name != value or ".." in path.parts for path in paths
    ):
        raise ValueError(f"Model basename '{value}' is not safe.")
