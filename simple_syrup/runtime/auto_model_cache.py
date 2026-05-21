# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist resolved automatic model paths under the ComfyUI user directory."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
CACHE_VERSION = 1
CacheSource = Literal["cached", "found", "downloaded"]


class AutoModelCacheError(ValueError):
    """Raised when persisted automatic model cache data is invalid."""


@dataclass(frozen=True)
class AutoModelCacheEntry:
    """A remembered automatic model artifact resolution."""

    folder_name: str
    filename: str
    path: Path
    source: CacheSource
    sha256: str

    def to_payload(self) -> dict[str, str]:
        """Return this cache entry as a JSON-serializable payload."""

        return {
            "folder_name": self.folder_name,
            "filename": self.filename,
            "path": str(self.path),
            "source": self.source,
            "sha256": self.sha256,
        }

    @classmethod
    def from_payload(
        cls,
        cache_id: str,
        payload: object,
    ) -> AutoModelCacheEntry:
        """Create a cache entry from validated JSON-like data."""

        if not isinstance(payload, dict):
            raise AutoModelCacheError(
                f"Auto model cache entry '{cache_id}' must be an object."
            )
        folder_name = _required_string(payload, cache_id, "folder_name")
        filename = _required_string(payload, cache_id, "filename")
        path = _required_string(payload, cache_id, "path")
        source = _required_source(payload, cache_id)
        sha256 = _required_string(payload, cache_id, "sha256")
        return cls(
            folder_name=folder_name,
            filename=filename,
            path=Path(path),
            source=source,
            sha256=sha256,
        )


class AutoModelCache:
    """Load and save automatic model resolutions in ComfyUI user storage."""

    def __init__(
        self,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a cache repository with injectable ComfyUI folder paths."""

        self._folder_paths_module = folder_paths_module

    def load(self) -> dict[str, AutoModelCacheEntry]:
        """Return every valid cache entry, or an empty cache when absent."""

        path = self.cache_path()
        if not path.is_file():
            return {}
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise AutoModelCacheError(
                f"Auto model cache at '{path}' is not valid JSON."
            ) from error
        return _entries_from_payload(payload)

    def save(self, entries: dict[str, AutoModelCacheEntry]) -> None:
        """Persist all cache entries atomically."""

        path = self.cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "version": CACHE_VERSION,
            "entries": {
                cache_id: entry.to_payload()
                for cache_id, entry in sorted(entries.items())
            },
        }
        temporary_path = path.with_name(f"{path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def save_entry(self, cache_id: str, entry: AutoModelCacheEntry) -> None:
        """Upsert one cache entry while preserving unrelated entries."""

        entries = self.load()
        entries[cache_id] = entry
        self.save(entries)
        LOGGER.info(
            "auto model cache entry saved",
            extra={
                "cache_id": cache_id,
                "folder_name": entry.folder_name,
                "path": str(entry.path),
                "source": entry.source,
            },
        )

    def cache_path(self) -> Path:
        """Return the JSON cache path under ComfyUI's user directory."""

        return (
            _user_directory(self._folder_paths()) / "simple_syrup" / "auto_models.json"
        )

    def _folder_paths(self) -> ModuleType:
        """Return the ComfyUI folder_paths module."""

        if self._folder_paths_module is not None:
            return self._folder_paths_module
        module: Any = importlib.import_module("folder_paths")
        if not isinstance(module, ModuleType):
            raise TypeError("folder_paths import did not return a module.")
        self._folder_paths_module = module
        return module


def _entries_from_payload(payload: object) -> dict[str, AutoModelCacheEntry]:
    """Parse cache entries from a JSON-like object."""

    if not isinstance(payload, dict):
        raise AutoModelCacheError("Auto model cache root must be an object.")
    version = payload.get("version")
    if version != CACHE_VERSION:
        raise AutoModelCacheError(f"Auto model cache version must be {CACHE_VERSION}.")
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, dict):
        raise AutoModelCacheError("Auto model cache entries must be an object.")

    entries: dict[str, AutoModelCacheEntry] = {}
    for cache_id, entry_payload in entries_payload.items():
        if not isinstance(cache_id, str):
            raise AutoModelCacheError("Auto model cache entry keys must be strings.")
        entries[cache_id] = AutoModelCacheEntry.from_payload(
            cache_id,
            entry_payload,
        )
    return entries


def _required_string(
    payload: dict[Any, Any],
    cache_id: str,
    field_name: str,
) -> str:
    """Return one required string field from a cache entry payload."""

    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise AutoModelCacheError(
            f"Auto model cache entry '{cache_id}' field '{field_name}' "
            "must be a non-empty string."
        )
    return value


def _required_source(payload: dict[Any, Any], cache_id: str) -> CacheSource:
    """Return the validated source field for a cache entry payload."""

    value = _required_string(payload, cache_id, "source")
    if value not in ("cached", "found", "downloaded"):
        raise AutoModelCacheError(
            f"Auto model cache entry '{cache_id}' source is invalid."
        )
    return cast(CacheSource, value)


def _user_directory(folder_paths: ModuleType) -> Path:
    """Return ComfyUI's user directory using the host's current conventions."""

    get_user_directory = getattr(folder_paths, "get_user_directory", None)
    if callable(get_user_directory):
        return Path(str(get_user_directory()))

    user_directory_attribute = getattr(folder_paths, "user_directory", None)
    if user_directory_attribute is not None:
        return Path(str(user_directory_attribute))

    models_dir: Any = folder_paths.models_dir
    return Path(str(models_dir)).parent / "user"
