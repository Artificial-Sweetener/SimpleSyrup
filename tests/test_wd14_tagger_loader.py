# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for WD14 tagger model loading and caching."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from simple_syrup.runtime.loaded_models import LoadedWD14Tagger
from simple_syrup.runtime.model_downloads import (
    DownloadRequest,
    DownloadResult,
    ProgressReporter,
)
from simple_syrup.runtime.wd14_tagger import FloatArray, WD14TagRecord
from simple_syrup.runtime.wd14_tagger_loader import (
    WD14TaggerCacheKey,
    WD14TaggerLoaderService,
)
from test_helpers import FakeFolderPaths


def test_loader_reuses_existing_files_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing ONNX and CSV artifacts are loaded without download requests."""

    _install_onnxruntime(monkeypatch, ("CPUExecutionProvider",))
    _create_wd14_files(tmp_path, "wd-eva02-large-tagger-v3")
    downloader = _FakeDownloader()
    session_factory = _SessionFactory()

    loaded = WD14TaggerLoaderService(
        downloader=downloader,
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=session_factory,
        cache={},
    ).load_model("wd-eva02-large-tagger-v3", auto_download=True)

    assert loaded.model_id == "wd-eva02-large-tagger-v3"
    assert loaded.source == "local"
    assert loaded.providers == ("CPUExecutionProvider",)
    assert loaded.tags == (WD14TagRecord("blue_hair", "0"),)
    assert downloader.requests == []
    assert session_factory.calls == [
        (tmp_path / "wd14_tagger" / "wd-eva02-large-tagger-v3.onnx", loaded.providers)
    ]


def test_loader_downloads_missing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing WD14 files download through the shared model downloader boundary."""

    _install_onnxruntime(monkeypatch, ("CPUExecutionProvider",))
    downloader = _FakeDownloader()

    loaded = WD14TaggerLoaderService(
        downloader=downloader,
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=_SessionFactory(),
        cache={},
    ).load_model("wd-eva02-large-tagger-v3", auto_download=True)

    assert loaded.onnx_path.is_file()
    assert loaded.csv_path.is_file()
    assert loaded.source == "downloaded: SmilingWolf/wd-eva02-large-tagger-v3"
    assert [request.destination_path.name for request in downloader.requests] == [
        "wd-eva02-large-tagger-v3.onnx",
        "wd-eva02-large-tagger-v3.csv",
    ]


def test_loader_rejects_unknown_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown WD14 selections fail before filesystem or ONNX work."""

    _install_onnxruntime(monkeypatch, ("CPUExecutionProvider",))
    service = WD14TaggerLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=_SessionFactory(),
        cache={},
    )

    with pytest.raises(ValueError, match="Unknown WD14 tagger model"):
        service.load_model("not-a-model", auto_download=True)


def test_loader_fails_when_missing_and_download_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing WD14 files fail clearly when auto-download is disabled."""

    _install_onnxruntime(monkeypatch, ("CPUExecutionProvider",))
    service = WD14TaggerLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=_SessionFactory(),
        cache={},
    )

    with pytest.raises(FileNotFoundError, match="auto_download is disabled"):
        service.load_model("wd-eva02-large-tagger-v3", auto_download=False)


def test_loader_uses_process_cache_for_identical_resolved_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical WD14 loads reuse the same loaded container and ONNX session."""

    _install_onnxruntime(monkeypatch, ("CPUExecutionProvider",))
    _create_wd14_files(tmp_path, "wd-eva02-large-tagger-v3")
    cache: dict[WD14TaggerCacheKey, LoadedWD14Tagger] = {}
    session_factory = _SessionFactory()
    service = WD14TaggerLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=session_factory,
        cache=cache,
    )

    first = service.load_model("wd-eva02-large-tagger-v3", auto_download=True)
    second = service.load_model("wd-eva02-large-tagger-v3", auto_download=True)

    assert second is first
    assert len(session_factory.calls) == 1


def test_loader_cache_separates_provider_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider changes produce distinct cached WD14 loaded containers."""

    _install_onnxruntime(monkeypatch, ("CPUExecutionProvider", "CUDAExecutionProvider"))
    _create_wd14_files(tmp_path, "wd-eva02-large-tagger-v3")
    cache: dict[WD14TaggerCacheKey, LoadedWD14Tagger] = {}
    first = WD14TaggerLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=_SessionFactory(),
        providers=("CPUExecutionProvider",),
        cache=cache,
    ).load_model("wd-eva02-large-tagger-v3", auto_download=True)
    second = WD14TaggerLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=_SessionFactory(),
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        cache=cache,
    ).load_model("wd-eva02-large-tagger-v3", auto_download=True)

    assert second is not first
    assert len(cache) == 2


def test_loader_does_not_cache_failed_session_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed WD14 session load leaves the cache empty for retry."""

    _install_onnxruntime(monkeypatch, ("CPUExecutionProvider",))
    _create_wd14_files(tmp_path, "wd-eva02-large-tagger-v3")
    cache: dict[WD14TaggerCacheKey, LoadedWD14Tagger] = {}
    session_factory = _FailOnceSessionFactory()
    service = WD14TaggerLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=session_factory,
        cache=cache,
    )

    with pytest.raises(RuntimeError, match="session failed"):
        service.load_model("wd-eva02-large-tagger-v3", auto_download=True)

    loaded = service.load_model("wd-eva02-large-tagger-v3", auto_download=True)

    assert isinstance(loaded, LoadedWD14Tagger)
    assert len(session_factory.calls) == 2
    assert len(cache) == 1


def test_loader_falls_back_to_available_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable requested ONNX providers are filtered before session loading."""

    _install_onnxruntime(monkeypatch, ("CPUExecutionProvider",))
    _create_wd14_files(tmp_path, "wd-eva02-large-tagger-v3")

    loaded = WD14TaggerLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        session_factory=_SessionFactory(),
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        cache={},
    ).load_model("wd-eva02-large-tagger-v3", auto_download=True)

    assert loaded.providers == ("CPUExecutionProvider",)


def test_loader_fails_when_no_provider_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider selection fails clearly when ONNX Runtime offers no usable provider."""

    _install_onnxruntime(monkeypatch, ())
    _create_wd14_files(tmp_path, "wd-eva02-large-tagger-v3")

    with pytest.raises(RuntimeError, match="No requested ONNX Runtime providers"):
        WD14TaggerLoaderService(
            folder_paths_module=FakeFolderPaths(tmp_path),
            session_factory=_SessionFactory(),
            providers=("CUDAExecutionProvider",),
            cache={},
        ).load_model("wd-eva02-large-tagger-v3", auto_download=True)


class _FakeDownloader:
    """Fake shared downloader that writes requested WD14 artifacts."""

    def __init__(self) -> None:
        """Initialize captured requests."""

        self.requests: list[DownloadRequest] = []

    def download(
        self,
        request: DownloadRequest,
        progress: ProgressReporter | None = None,
    ) -> DownloadResult:
        """Record and satisfy a trusted download request."""

        _ = progress
        self.requests.append(request)
        request.destination_path.parent.mkdir(parents=True, exist_ok=True)
        if request.destination_path.suffix == ".csv":
            request.destination_path.write_text(
                "id,name,category\n0,blue_hair,0\n",
                encoding="utf-8",
            )
        else:
            request.destination_path.write_bytes(b"onnx")
        return DownloadResult(
            path=request.destination_path,
            bytes_downloaded=1,
            skipped_existing=False,
        )


class _SessionFactory:
    """Record ONNX session creation calls."""

    def __init__(self) -> None:
        """Initialize captured calls."""

        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def __call__(self, path: Path, providers: tuple[str, ...]) -> _FakeSession:
        """Return a fake WD14 session."""

        self.calls.append((path, providers))
        return _FakeSession()


class _FailOnceSessionFactory(_SessionFactory):
    """Fail the first ONNX session creation and succeed afterward."""

    def __call__(self, path: Path, providers: tuple[str, ...]) -> _FakeSession:
        """Record each call and fail only the first one."""

        self.calls.append((path, providers))
        if len(self.calls) == 1:
            raise RuntimeError("session failed")
        return _FakeSession()


class _FakeSession:
    """Minimal WD14 session double."""

    def get_inputs(self) -> list[object]:
        """Return no fake inputs."""

        return []

    def get_outputs(self) -> list[object]:
        """Return no fake outputs."""

        return []

    def run(
        self, output_names: list[str], feeds: dict[str, FloatArray]
    ) -> list[object]:
        """Return no fake outputs."""

        _ = output_names, feeds
        return []


def _create_wd14_files(tmp_path: Path, model_id: str) -> None:
    """Create a complete local WD14 ONNX/CSV pair."""

    model_dir = tmp_path / "wd14_tagger"
    model_dir.mkdir(parents=True)
    (model_dir / f"{model_id}.onnx").write_bytes(b"onnx")
    (model_dir / f"{model_id}.csv").write_text(
        "id,name,category\n0,blue_hair,0\n",
        encoding="utf-8",
    )


def _install_onnxruntime(
    monkeypatch: pytest.MonkeyPatch,
    providers: tuple[str, ...],
) -> None:
    """Install a minimal fake onnxruntime module."""

    module = ModuleType("onnxruntime")
    module.get_available_providers = lambda: list(providers)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
