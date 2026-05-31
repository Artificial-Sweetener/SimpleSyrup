# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistent backend settings for SimpleSyrup runtime behavior."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from ..domain.external_llm import (
    ExternalLLMConfigError,
    normalize_base_url,
    normalize_model_ids,
)
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
SETTINGS_FILENAME: Final = "settings.json"


class SimpleSyrupSettingsError(ValueError):
    """Raised when SimpleSyrup settings data is malformed."""


@dataclass(frozen=True)
class ExternalLLMSettings:
    """Non-secret external LLM provider settings persisted in Comfy's user data."""

    base_url: str = ""
    cached_models: tuple[str, ...] = ()
    default_model: str = ""

    def to_payload(self) -> dict[str, object]:
        """Return the validated external LLM JSON payload shape."""

        return {
            "base_url": self.base_url,
            "cached_models": list(self.cached_models),
            "default_model": self.default_model,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ExternalLLMSettings:
        """Create external LLM settings from a decoded JSON payload."""

        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup external_llm settings must be a JSON object."
            )

        base_url = payload.get("base_url", "")
        if not isinstance(base_url, str):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup external_llm.base_url must be a string."
            )

        default_model = payload.get("default_model", "")
        if not isinstance(default_model, str):
            raise SimpleSyrupSettingsError(
                "SimpleSyrup external_llm.default_model must be a string."
            )

        try:
            cached_models = normalize_model_ids(payload.get("cached_models", []))
        except ExternalLLMConfigError as error:
            raise SimpleSyrupSettingsError(str(error)) from error

        default = default_model.strip()
        if default and cached_models and default not in cached_models:
            default = cached_models[0]

        normalized_url = ""
        if base_url.strip():
            try:
                normalized_url = normalize_base_url(base_url)
            except ExternalLLMConfigError as error:
                raise SimpleSyrupSettingsError(str(error)) from error

        return cls(
            base_url=normalized_url,
            cached_models=cached_models,
            default_model=default,
        )


@dataclass(frozen=True)
class SimpleSyrupSettings:
    """User-configurable SimpleSyrup runtime settings."""

    show_downloadable_models: bool = True
    external_llm: ExternalLLMSettings = field(default_factory=ExternalLLMSettings)

    def to_payload(self) -> dict[str, object]:
        """Return the validated JSON payload shape."""

        return {
            "show_downloadable_models": self.show_downloadable_models,
            "external_llm": self.external_llm.to_payload(),
        }

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

        try:
            external_llm = ExternalLLMSettings.from_payload(payload.get("external_llm"))
        except SimpleSyrupSettingsError as error:
            LOGGER.warning(
                "using default external llm settings after failed load",
                extra={"reason": str(error)},
            )
            external_llm = ExternalLLMSettings()

        return cls(show_downloadable_models=value, external_llm=external_llm)


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
