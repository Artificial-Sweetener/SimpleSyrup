# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistent backend settings for SimpleSyrup runtime behavior."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
SETTINGS_FILENAME: Final = "settings.json"


class SimpleSyrupSettingsError(ValueError):
    """Raised when SimpleSyrup settings data is malformed."""


@dataclass(frozen=True)
class SimpleSyrupSettings:
    """User-configurable SimpleSyrup runtime settings."""

    show_downloadable_models: bool = True

    def to_payload(self) -> dict[str, bool]:
        """Return the validated JSON payload shape."""

        return {"show_downloadable_models": self.show_downloadable_models}

    @classmethod
    def from_payload(cls, payload: object) -> SimpleSyrupSettings:
        """Create settings from a decoded JSON payload."""

        if not isinstance(payload, dict):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup settings payload must be a JSON object."
            )

        value = payload.get("show_downloadable_models")
        if not isinstance(value, bool):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup settings payload is invalid. Expected "
                "show_downloadable_models to be a boolean."
            )
        return cls(show_downloadable_models=value)


class SimpleSyrupSettingsRepository:
    """Load and save SimpleSyrup settings from Comfy's user directory."""

    def __init__(
        self,
        settings_path: Path | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a repository with injectable filesystem and Comfy boundaries."""

        self._settings_path = settings_path
        self._folder_paths_module = folder_paths_module

    def load(self) -> SimpleSyrupSettings:
        """Load settings or return defaults for missing/malformed files."""

        path = self.settings_path()
        if not path.is_file():
            return SimpleSyrupSettings()

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return SimpleSyrupSettings.from_payload(payload)
        except (JSONDecodeError, OSError, SimpleSyrupSettingsError) as error:
            LOGGER.warning(
                "using default settings after failed load",
                extra={"settings_path": str(path), "reason": str(error)},
            )
            return SimpleSyrupSettings()

    def save(self, settings: SimpleSyrupSettings) -> SimpleSyrupSettings:
        """Persist validated settings and return the saved value."""

        path = self.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        temporary_path.write_text(
            json.dumps(settings.to_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return settings

    def settings_path(self) -> Path:
        """Return the resolved settings path."""

        if self._settings_path is not None:
            return self._settings_path

        folder_paths = self._folder_paths_module or _folder_paths()
        return (
            _user_directory(folder_paths)
            / "default"
            / "SimpleSyrup"
            / SETTINGS_FILENAME
        )


def _user_directory(folder_paths: ModuleType) -> Path:
    """Return Comfy's user directory from stable APIs or conservative fallback."""

    get_user_directory = getattr(folder_paths, "get_user_directory", None)
    if callable(get_user_directory):
        user_directory = get_user_directory()
        return Path(str(user_directory))

    user_directory_attribute = getattr(folder_paths, "user_directory", None)
    if user_directory_attribute is not None:
        return Path(str(user_directory_attribute))

    models_dir: Any = folder_paths.models_dir
    return Path(str(models_dir)).parent / "user"


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module
