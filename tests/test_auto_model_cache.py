# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for persisted automatic model path cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_syrup.runtime.auto_model_cache import (
    AutoModelCache,
    AutoModelCacheEntry,
    AutoModelCacheError,
)
from test_helpers import FakeFolderPaths


def test_cache_uses_comfy_user_directory(tmp_path: Path) -> None:
    """Cache path is rooted in ComfyUI's user directory when available."""

    fake = FakeFolderPaths(tmp_path / "models")
    fake.get_user_directory = lambda: str(tmp_path / "custom_user")  # type: ignore[attr-defined]

    cache = AutoModelCache(fake)

    assert cache.cache_path() == (
        tmp_path / "custom_user" / "simple_syrup" / "auto_models.json"
    )


def test_cache_falls_back_to_user_next_to_models(tmp_path: Path) -> None:
    """Cache path falls back beside models when ComfyUI lacks a helper."""

    fake = FakeFolderPaths(tmp_path / "models")

    cache = AutoModelCache(fake)

    assert cache.cache_path() == tmp_path / "user" / "simple_syrup" / "auto_models.json"


def test_missing_cache_loads_empty_entries(tmp_path: Path) -> None:
    """A missing cache file is treated as an empty cache."""

    assert AutoModelCache(FakeFolderPaths(tmp_path / "models")).load() == {}


def test_cache_saves_and_loads_entry(tmp_path: Path) -> None:
    """Cache entries round-trip through JSON persistence."""

    cache = AutoModelCache(FakeFolderPaths(tmp_path / "models"))
    entry = AutoModelCacheEntry(
        folder_name="text_encoders",
        filename="qwen_3_06b_base.safetensors",
        path=tmp_path
        / "models"
        / "text_encoders"
        / "qwen"
        / "qwen_3_06b_base.safetensors",
        source="downloaded",
        sha256="abc123",
    )

    cache.save_entry("anima_qwen_text_encoder", entry)

    assert cache.load() == {"anima_qwen_text_encoder": entry}


def test_cache_preserves_unrelated_entries(tmp_path: Path) -> None:
    """Updating one entry does not discard other cached artifacts."""

    cache = AutoModelCache(FakeFolderPaths(tmp_path / "models"))
    first = AutoModelCacheEntry(
        folder_name="text_encoders",
        filename="first.safetensors",
        path=tmp_path / "first.safetensors",
        source="found",
        sha256="first",
    )
    second = AutoModelCacheEntry(
        folder_name="vae",
        filename="second.safetensors",
        path=tmp_path / "second.safetensors",
        source="downloaded",
        sha256="second",
    )

    cache.save_entry("first", first)
    cache.save_entry("second", second)

    assert cache.load() == {"first": first, "second": second}


def test_cache_rejects_invalid_json_structure(tmp_path: Path) -> None:
    """Invalid cache schema fails with actionable context."""

    cache = AutoModelCache(FakeFolderPaths(tmp_path / "models"))
    cache.cache_path().parent.mkdir(parents=True)
    cache.cache_path().write_text(json.dumps({"entries": []}), encoding="utf-8")

    with pytest.raises(AutoModelCacheError, match="version"):
        cache.load()


def test_cache_rejects_invalid_entry_field(tmp_path: Path) -> None:
    """Invalid cache entry fields are rejected during load."""

    cache = AutoModelCache(FakeFolderPaths(tmp_path / "models"))
    cache.cache_path().parent.mkdir(parents=True)
    cache.cache_path().write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    "bad": {
                        "folder_name": "vae",
                        "filename": "model.safetensors",
                        "path": "",
                        "source": "found",
                        "sha256": "abc",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AutoModelCacheError, match="path"):
        cache.load()
