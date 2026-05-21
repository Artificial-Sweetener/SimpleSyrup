# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ViTMatte loader runtime service."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from simple_syrup.runtime.loaded_models import LoadedViTMatteModel
from simple_syrup.runtime.model_catalog import get_vitmatte_entry, vitmatte_choices
from simple_syrup.runtime.vitmatte_loader import (
    ViTMatteLoaderService,
    ViTMatteModelCacheKey,
    is_valid_vitmatte_directory,
)
from test_helpers import FakeFolderPaths


class RecordingSnapshotDownloader:
    """Downloader double that writes a valid ViTMatte snapshot."""

    def __init__(self) -> None:
        """Create an empty downloader recorder."""

        self.requests: list[tuple[str, Path]] = []

    def download_snapshot(
        self,
        repo_id: str,
        destination: Path,
        progress: object | None = None,
    ) -> Path:
        """Record and satisfy a snapshot download request."""

        self.requests.append((repo_id, destination))
        _write_vitmatte_snapshot(destination)
        return destination


def test_vitmatte_choices_include_small_and_base() -> None:
    """ViTMatte catalog exposes the intended model choices."""

    assert vitmatte_choices() == [
        "vitmatte-small-composition-1k",
        "vitmatte-base-composition-1k",
    ]


def test_vitmatte_valid_directory_requires_expected_files(tmp_path: Path) -> None:
    """ViTMatte directory validation is bounded to expected HF snapshot files."""

    _write_vitmatte_snapshot(tmp_path)

    assert is_valid_vitmatte_directory(tmp_path) is True


def test_vitmatte_loader_prefers_canonical_path(tmp_path: Path) -> None:
    """Canonical SimpleSyrup paths win over LayerStyle-compatible paths."""

    canonical = tmp_path / "vitmatte" / "vitmatte-small-composition-1k"
    layerstyle = tmp_path / "vitmatte"
    _write_vitmatte_snapshot(canonical)
    _write_vitmatte_snapshot(layerstyle)

    resolved = ViTMatteLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path)
    ).resolve_model_directory(
        get_vitmatte_entry("vitmatte-small-composition-1k"),
        auto_download=False,
    )

    assert resolved.path == canonical


def test_vitmatte_loader_reuses_layerstyle_small_path(tmp_path: Path) -> None:
    """LayerStyle's small-model path is reused when it is a valid snapshot."""

    layerstyle = tmp_path / "vitmatte"
    _write_vitmatte_snapshot(layerstyle)

    resolved = ViTMatteLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path)
    ).resolve_model_directory(
        get_vitmatte_entry("vitmatte-small-composition-1k"),
        auto_download=False,
    )

    assert resolved.path == layerstyle
    assert resolved.source == "layerstyle-compatible"


def test_vitmatte_loader_reuses_layerstyle_base_path(tmp_path: Path) -> None:
    """LayerStyle's base-model path is reused when it is a valid snapshot."""

    layerstyle = tmp_path / "vitmatte-base-composition-1k"
    _write_vitmatte_snapshot(layerstyle)

    resolved = ViTMatteLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path)
    ).resolve_model_directory(
        get_vitmatte_entry("vitmatte-base-composition-1k"),
        auto_download=False,
    )

    assert resolved.path == layerstyle


def test_vitmatte_loader_downloads_missing_model_to_canonical_path(
    tmp_path: Path,
) -> None:
    """Missing ViTMatte models download to SimpleSyrup's canonical layout."""

    downloader = RecordingSnapshotDownloader()
    resolved = ViTMatteLoaderService(
        downloader=downloader,
        folder_paths_module=FakeFolderPaths(tmp_path),
    ).resolve_model_directory(
        get_vitmatte_entry("vitmatte-small-composition-1k"),
        auto_download=True,
    )

    assert resolved.path == tmp_path / "vitmatte" / "vitmatte-small-composition-1k"
    assert downloader.requests == [
        (
            "hustvl/vitmatte-small-composition-1k",
            tmp_path / "vitmatte" / "vitmatte-small-composition-1k",
        )
    ]


def test_vitmatte_loader_fails_when_missing_and_download_disabled(
    tmp_path: Path,
) -> None:
    """Missing ViTMatte models fail clearly when auto-download is disabled."""

    with pytest.raises(FileNotFoundError, match="auto_download is disabled"):
        ViTMatteLoaderService(
            folder_paths_module=FakeFolderPaths(tmp_path)
        ).resolve_model_directory(
            get_vitmatte_entry("vitmatte-small-composition-1k"),
            auto_download=False,
        )


def test_vitmatte_loader_loads_transformers_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ViTMatte loader loads model and processor from a valid local directory."""

    _write_vitmatte_snapshot(tmp_path / "vitmatte")
    _install_fake_transformers(monkeypatch)

    loaded = ViTMatteLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path)
    ).load_model("vitmatte-small-composition-1k", auto_download=False)

    assert isinstance(loaded, LoadedViTMatteModel)
    assert loaded.model_id == "vitmatte-small-composition-1k"
    assert loaded.managed_model is not None


def test_vitmatte_loader_uses_process_cache_for_identical_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical ViTMatte loads reuse the same model and processor container."""

    _write_vitmatte_snapshot(tmp_path / "vitmatte")
    state = _install_fake_transformers(monkeypatch)
    cache: dict[ViTMatteModelCacheKey, LoadedViTMatteModel] = {}
    service = ViTMatteLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    first = service.load_model("vitmatte-small-composition-1k", auto_download=False)
    second = service.load_model("vitmatte-small-composition-1k", auto_download=False)

    assert second is first
    assert state.model_paths == [str(tmp_path / "vitmatte")]
    assert state.processor_paths == [str(tmp_path / "vitmatte")]
    assert len(cache) == 1


def test_vitmatte_loader_cache_separates_model_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different ViTMatte model paths produce separate loaded containers."""

    _write_vitmatte_snapshot(tmp_path / "vitmatte")
    _write_vitmatte_snapshot(tmp_path / "vitmatte-base-composition-1k")
    state = _install_fake_transformers(monkeypatch)
    cache: dict[ViTMatteModelCacheKey, LoadedViTMatteModel] = {}
    service = ViTMatteLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    first = service.load_model("vitmatte-small-composition-1k", auto_download=False)
    second = service.load_model("vitmatte-base-composition-1k", auto_download=False)

    assert second is not first
    assert state.model_paths == [
        str(tmp_path / "vitmatte"),
        str(tmp_path / "vitmatte-base-composition-1k"),
    ]
    assert len(cache) == 2


def test_vitmatte_loader_does_not_cache_failed_transformers_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ViTMatte transformers load leaves the cache empty for retry."""

    _write_vitmatte_snapshot(tmp_path / "vitmatte")
    state = _install_fake_transformers(monkeypatch, fail_once=True)
    cache: dict[ViTMatteModelCacheKey, LoadedViTMatteModel] = {}
    service = ViTMatteLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    with pytest.raises(RuntimeError, match="ViTMatte failed"):
        service.load_model("vitmatte-small-composition-1k", auto_download=False)

    loaded = service.load_model("vitmatte-small-composition-1k", auto_download=False)

    assert isinstance(loaded, LoadedViTMatteModel)
    assert len(state.model_paths) == 2
    assert len(cache) == 1


def _write_vitmatte_snapshot(path: Path) -> None:
    """Write a minimal valid ViTMatte directory."""

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


@dataclass
class _FakeTransformersState:
    """Record fake Transformers ViTMatte loads."""

    model_paths: list[str] = field(default_factory=list)
    processor_paths: list[str] = field(default_factory=list)
    fail_once: bool = False


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    fail_once: bool = False,
) -> _FakeTransformersState:
    """Install fake transformers ViTMatte classes."""

    state = _FakeTransformersState(fail_once=fail_once)

    class FakeModel:
        """Minimal model fake."""

        @classmethod
        def from_pretrained(
            cls,
            path: str,
            local_files_only: bool,
        ) -> FakeModel:
            """Return a fake model."""

            assert local_files_only is True
            state.model_paths.append(path)
            if state.fail_once:
                state.fail_once = False
                raise RuntimeError("ViTMatte failed")
            return cls()

        def eval(self) -> None:
            """Accept eval mode."""

    class FakeProcessor:
        """Minimal processor fake."""

        @classmethod
        def from_pretrained(
            cls,
            path: str,
            local_files_only: bool,
        ) -> FakeProcessor:
            """Return a fake processor."""

            assert local_files_only is True
            state.processor_paths.append(path)
            return cls()

    module = ModuleType("transformers")
    module.VitMatteForImageMatting = FakeModel  # type: ignore[attr-defined]
    module.VitMatteImageProcessor = FakeProcessor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    return state
