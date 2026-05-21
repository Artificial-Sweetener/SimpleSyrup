# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SimpleSyrup backend settings persistence."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

from simple_syrup.runtime.settings import (
    SimpleSyrupSettings,
    SimpleSyrupSettingsError,
    SimpleSyrupSettingsRepository,
)


def test_default_settings_show_downloadable_models() -> None:
    """Default settings favor low-friction model discovery."""

    assert SimpleSyrupSettings().show_downloadable_models is True


def test_missing_settings_file_returns_defaults(tmp_path: Path) -> None:
    """Missing persisted settings are treated as default settings."""

    repository = SimpleSyrupSettingsRepository(tmp_path / "settings.json")

    assert repository.load() == SimpleSyrupSettings()


def test_valid_settings_file_is_loaded(tmp_path: Path) -> None:
    """A valid settings file controls the backend setting."""

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"show_downloadable_models": False}),
        encoding="utf-8",
    )

    assert SimpleSyrupSettingsRepository(path).load() == SimpleSyrupSettings(
        show_downloadable_models=False
    )


def test_invalid_json_file_returns_defaults(tmp_path: Path) -> None:
    """Malformed JSON fails closed to defaults without deleting user data."""

    path = tmp_path / "settings.json"
    path.write_text("{not-json", encoding="utf-8")

    assert SimpleSyrupSettingsRepository(path).load() == SimpleSyrupSettings()
    assert path.read_text(encoding="utf-8") == "{not-json"


def test_invalid_schema_returns_defaults(tmp_path: Path) -> None:
    """Malformed settings schema fails closed to defaults."""

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"show_downloadable_models": "false"}),
        encoding="utf-8",
    )

    assert SimpleSyrupSettingsRepository(path).load() == SimpleSyrupSettings()


def test_saving_settings_writes_validated_schema(tmp_path: Path) -> None:
    """Saving settings writes only the known schema."""

    path = tmp_path / "nested" / "settings.json"
    repository = SimpleSyrupSettingsRepository(path)

    repository.save(SimpleSyrupSettings(show_downloadable_models=False))

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "show_downloadable_models": False
    }


def test_settings_path_uses_comfy_user_directory(tmp_path: Path) -> None:
    """Settings path resolution uses Comfy's user directory API."""

    folder_paths = ModuleType("folder_paths")
    folder_paths.get_user_directory = lambda: str(tmp_path)  # type: ignore[attr-defined]

    repository = SimpleSyrupSettingsRepository(folder_paths_module=folder_paths)

    assert repository.settings_path() == (
        tmp_path / "default" / "SimpleSyrup" / "settings.json"
    )


def test_payload_validation_rejects_non_boolean_value() -> None:
    """Schema validation rejects non-boolean setting values."""

    with pytest.raises(SimpleSyrupSettingsError, match="show_downloadable_models"):
        SimpleSyrupSettings.from_payload({"show_downloadable_models": 1})
