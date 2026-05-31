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
        "external_llm": {
            "base_url": "",
            "cached_models": [],
            "default_model": "",
        },
        "show_downloadable_models": False,
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


def test_missing_external_llm_settings_loads_defaults(tmp_path: Path) -> None:
    """Existing settings files without external LLM settings remain valid."""

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"show_downloadable_models": False}),
        encoding="utf-8",
    )

    settings = SimpleSyrupSettingsRepository(path).load()

    assert settings.show_downloadable_models is False
    assert settings.external_llm.base_url == ""
    assert settings.external_llm.cached_models == ()
    assert settings.external_llm.default_model == ""


def test_valid_external_llm_settings_are_loaded(tmp_path: Path) -> None:
    """External LLM settings are normalized when loaded."""

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "show_downloadable_models": True,
                "external_llm": {
                    "base_url": "https://provider.example/v1/",
                    "cached_models": ["model-a", "model-a", "model-b"],
                    "default_model": "model-b",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = SimpleSyrupSettingsRepository(path).load()

    assert settings.external_llm.base_url == "https://provider.example/v1"
    assert settings.external_llm.cached_models == ("model-a", "model-b")
    assert settings.external_llm.default_model == "model-b"


def test_invalid_external_llm_settings_fall_back_to_external_defaults(
    tmp_path: Path,
) -> None:
    """Malformed external LLM settings do not invalidate other settings."""

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "show_downloadable_models": False,
                "external_llm": {
                    "base_url": "not-a-url",
                    "cached_models": ["model-a"],
                    "default_model": "model-a",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = SimpleSyrupSettingsRepository(path).load()

    assert settings.show_downloadable_models is False
    assert settings.external_llm.base_url == ""
