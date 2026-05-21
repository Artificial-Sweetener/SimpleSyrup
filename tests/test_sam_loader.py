# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SAM loader runtime service."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from simple_syrup.runtime.loaded_models import LoadedSAMModel
from simple_syrup.runtime.model_downloads import DownloadRequest, DownloadResult
from simple_syrup.runtime.sam_loader import SAMLoaderService, SAMModelCacheKey
from test_helpers import FakeFolderPaths


class RecordingDownloader:
    """Downloader double that writes requested artifacts."""

    def __init__(self) -> None:
        """Create an empty recording downloader."""

        self.requests: list[DownloadRequest] = []

    def download(
        self,
        request: DownloadRequest,
        progress: object | None = None,
    ) -> DownloadResult:
        """Record and satisfy a download request."""

        self.requests.append(request)
        request.destination_path.parent.mkdir(parents=True, exist_ok=True)
        request.destination_path.write_bytes(b"model")
        return DownloadResult(request.destination_path, 5, False)


def test_sam_loader_downloads_missing_known_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAM loader downloads known missing models when enabled."""

    downloader = RecordingDownloader()
    _install_fake_segment_anything(monkeypatch)

    loaded = SAMLoaderService(
        downloader=downloader,  # type: ignore[arg-type]
        folder_paths_module=FakeFolderPaths(tmp_path),
    ).load_model("sam_vit_b (375MB)", True)

    assert isinstance(loaded, LoadedSAMModel)
    assert loaded.managed_model is not None
    assert downloader.requests


def test_sam_loader_uses_process_cache_for_identical_resolved_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical SAM loads reuse the same loaded container and SAM model."""

    state = _install_fake_segment_anything(monkeypatch)
    _create_sam_file(tmp_path, "sam_vit_b_01ec64.pth")
    cache: dict[SAMModelCacheKey, LoadedSAMModel] = {}
    service = SAMLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    first = service.load_model("sam_vit_b (375MB)", auto_download=True)
    second = service.load_model("sam_vit_b (375MB)", auto_download=True)

    assert second is first
    assert state.checkpoints == [str(tmp_path / "sams" / "sam_vit_b_01ec64.pth")]
    assert len(cache) == 1


def test_sam_loader_cache_separates_model_selections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different SAM catalog selections produce separate loaded containers."""

    state = _install_fake_segment_anything(monkeypatch)
    _create_sam_file(tmp_path, "sam_vit_b_01ec64.pth")
    _create_sam_file(tmp_path, "sam_vit_l_0b3195.pth")
    cache: dict[SAMModelCacheKey, LoadedSAMModel] = {}
    service = SAMLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    first = service.load_model("sam_vit_b (375MB)", auto_download=True)
    second = service.load_model("sam_vit_l (1.25GB)", auto_download=True)

    assert second is not first
    assert state.checkpoints == [
        str(tmp_path / "sams" / "sam_vit_b_01ec64.pth"),
        str(tmp_path / "sams" / "sam_vit_l_0b3195.pth"),
    ]
    assert len(cache) == 2


def test_sam_loader_does_not_cache_failed_registry_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed SAM registry construction leaves the cache empty for retry."""

    state = _install_fake_segment_anything(monkeypatch, fail_once=True)
    _create_sam_file(tmp_path, "sam_vit_b_01ec64.pth")
    cache: dict[SAMModelCacheKey, LoadedSAMModel] = {}
    service = SAMLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    with pytest.raises(RuntimeError, match="SAM failed"):
        service.load_model("sam_vit_b (375MB)", auto_download=True)

    loaded = service.load_model("sam_vit_b (375MB)", auto_download=True)

    assert isinstance(loaded, LoadedSAMModel)
    assert len(state.checkpoints) == 2
    assert len(cache) == 1


def test_sam_loader_loads_sam_hq_from_owned_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAM-HQ catalog entries load from SimpleSyrup's vendored runtime."""

    state = _install_fake_sam_hq_runtime(monkeypatch)
    _create_sam_file(tmp_path, "sam_hq_vit_b.pth")

    loaded = SAMLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
    ).load_model("sam_hq_vit_b (379MB)", auto_download=True)

    assert isinstance(loaded, LoadedSAMModel)
    assert loaded.model_id == "sam_hq_vit_b"
    assert loaded.managed_model is not None
    assert state.checkpoints == [str(tmp_path / "sams" / "sam_hq_vit_b.pth")]


def test_sam_loader_errors_when_missing_and_download_disabled(tmp_path: Path) -> None:
    """SAM loader fails clearly when downloads are disabled."""

    with pytest.raises(FileNotFoundError, match="auto_download is disabled"):
        SAMLoaderService(folder_paths_module=FakeFolderPaths(tmp_path)).load_model(
            "sam_vit_b (375MB)",
            False,
        )


@dataclass
class _FakeSegmentAnythingState:
    """Record fake Segment Anything model construction."""

    checkpoints: list[str] = field(default_factory=list)
    fail_once: bool = False


def _install_fake_segment_anything(
    monkeypatch: pytest.MonkeyPatch,
    fail_once: bool = False,
) -> _FakeSegmentAnythingState:
    """Install a fake segment_anything registry for loader tests."""

    state = _FakeSegmentAnythingState(fail_once=fail_once)

    class FakeModel:
        """Minimal PyTorch-like model fake."""

        def __init__(self) -> None:
            """Create a model that records device movement."""

            self.to_calls = 0

        def to(self, device: object) -> None:
            """Record forbidden loader-time device movement."""

            self.to_calls += 1

        def eval(self) -> None:
            """Accept eval mode."""

    def build_model(checkpoint: str) -> FakeModel:
        """Record checkpoint construction and optionally fail once."""

        state.checkpoints.append(checkpoint)
        if state.fail_once:
            state.fail_once = False
            raise RuntimeError("SAM failed")
        return FakeModel()

    module = ModuleType("segment_anything")
    module.sam_model_registry = {"vit_b": build_model, "vit_l": build_model}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "segment_anything", module)
    return state


def _install_fake_sam_hq_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeSegmentAnythingState:
    """Install a fake SimpleSyrup SAM-HQ registry for loader tests."""

    state = _FakeSegmentAnythingState()

    class FakeModel:
        """Minimal SAM-HQ model fake."""

        def to(self, device: object) -> None:
            """Accept device movement."""

        def eval(self) -> None:
            """Accept eval mode."""

    def build_model(checkpoint: str) -> FakeModel:
        """Record SAM-HQ checkpoint construction."""

        state.checkpoints.append(checkpoint)
        return FakeModel()

    module = ModuleType("simple_syrup.third_party.sam_hq_runtime.build_sam_hq")
    module.sam_model_registry = {"sam_hq_vit_b": build_model}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return state


def _create_sam_file(tmp_path: Path, filename: str) -> None:
    """Create one local SAM checkpoint file."""

    model_dir = tmp_path / "sams"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / filename).write_bytes(b"model")
