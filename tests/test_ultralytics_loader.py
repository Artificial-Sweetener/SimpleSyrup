# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Ultralytics detector model loading."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from simple_syrup.runtime.model_catalog import get_ultralytics_entry
from simple_syrup.runtime.model_choices import ModelChoiceService
from simple_syrup.runtime.model_downloads import (
    DownloadRequest,
    DownloadResult,
    ModelDownloader,
    ProgressReporter,
)
from simple_syrup.runtime.settings import SimpleSyrupSettings
from simple_syrup.runtime.ultralytics_loader import (
    NO_LOCAL_ULTRALYTICS_MODELS,
    LoadedUltralyticsDetector,
    UltralyticsLoaderService,
    UltralyticsModelCacheKey,
)


def test_model_choices_list_conventional_folders(tmp_path: Path) -> None:
    """Model choices include root, bbox, and segmentation conventions."""

    models_dir = tmp_path / "models"
    (models_dir / "ultralytics").mkdir(parents=True)
    (models_dir / "ultralytics" / "bbox").mkdir()
    (models_dir / "ultralytics" / "segm").mkdir()
    (models_dir / "ultralytics" / "root.pt").write_bytes(b"")
    (models_dir / "ultralytics" / "bbox" / "face.pt").write_bytes(b"")
    (models_dir / "ultralytics" / "segm" / "person.pt").write_bytes(b"")

    folder_paths = _folder_paths(models_dir)
    service = UltralyticsLoaderService(
        folder_paths_module=folder_paths,
        choice_service=_choice_service(show_downloadable_models=False),
    )

    assert service.model_choices() == ["bbox/face.pt", "root.pt", "segm/person.pt"]


def test_model_choices_returns_sentinel_when_no_models(tmp_path: Path) -> None:
    """An empty model directory returns a clear dropdown sentinel."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    folder_paths = _folder_paths(models_dir)
    service = UltralyticsLoaderService(
        folder_paths_module=folder_paths,
        choice_service=_choice_service(show_downloadable_models=False),
    )

    assert service.model_choices() == [NO_LOCAL_ULTRALYTICS_MODELS]


def test_model_choices_include_curated_downloadable_models(tmp_path: Path) -> None:
    """Downloadable mode exposes the complete curated Anzhc model selection."""

    folder_paths = _folder_paths(tmp_path / "models")
    service = UltralyticsLoaderService(
        folder_paths_module=folder_paths,
        choice_service=_choice_service(show_downloadable_models=True),
    )

    choices = service.model_choices()

    assert len(choices) == 22
    assert "Anzhc Face -seg (6.52MB)" in choices
    assert "Bingsu Hand YOLOv8n (6.23MB)" in choices
    assert "Fuyucchi YOLOv8x6 Anime Face (195MB)" in choices
    assert "Anzhcs Breast size det cls v8 640 y11m (38.70MB)" not in choices
    assert not any("Drone" in choice for choice in choices)
    assert not any("Score" in choice for choice in choices)


def test_hidden_catalog_choices_exclude_installed_curated_model(
    tmp_path: Path,
) -> None:
    """Hidden catalog mode excludes installed curated model files."""

    models_dir = tmp_path / "models"
    checkpoint = models_dir / "ultralytics" / "segm" / "Anzhc Face -seg.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    folder_paths = _folder_paths(models_dir)
    service = UltralyticsLoaderService(
        folder_paths_module=folder_paths,
        choice_service=_choice_service(show_downloadable_models=False),
    )

    assert service.model_choices() == [NO_LOCAL_ULTRALYTICS_MODELS]


def test_missing_model_raises_value_error(tmp_path: Path) -> None:
    """Loading rejects unknown model choices before importing Ultralytics."""

    service = UltralyticsLoaderService(folder_paths_module=_folder_paths(tmp_path))

    with pytest.raises(ValueError, match="was not found"):
        service.resolve_model_path("bbox/missing.pt")


def test_missing_ultralytics_import_raises_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional Ultralytics dependency fails with install guidance."""

    models_dir = tmp_path / "models"
    (models_dir / "ultralytics").mkdir(parents=True)
    (models_dir / "ultralytics" / "model.pt").write_bytes(b"")

    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        if name == "ultralytics":
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    service = UltralyticsLoaderService(folder_paths_module=_folder_paths(models_dir))

    with pytest.raises(RuntimeError, match="requires the 'ultralytics' package"):
        service.load("model.pt")


def test_loader_returns_native_and_compatibility_outputs(tmp_path: Path) -> None:
    """Loading returns one native model and paired detector facades."""

    models_dir = tmp_path / "models"
    (models_dir / "ultralytics" / "segm").mkdir(parents=True)
    (models_dir / "ultralytics" / "segm" / "face.pt").write_bytes(b"")
    ultralytics_module = ModuleType("ultralytics")
    cast(Any, ultralytics_module).YOLO = _FakeYOLO

    service = UltralyticsLoaderService(
        folder_paths_module=_folder_paths(models_dir),
        ultralytics_module=ultralytics_module,
    )

    loaded = service.load("segm/face.pt")

    assert loaded.detector_model.model_name == "segm/face.pt"
    assert loaded.detector_model.supports_segmentation is True
    assert loaded.bbox_detector is cast(Any, loaded.segm_detector).bbox_detector


def test_curated_model_downloads_to_impact_pack_compatible_folder(
    tmp_path: Path,
) -> None:
    """A curated selection downloads with checksum verification into segm."""

    models_dir = tmp_path / "models"
    folder_paths = _folder_paths(models_dir)
    downloader = _RecordingDownloader()
    ultralytics_module = ModuleType("ultralytics")
    cast(Any, ultralytics_module).YOLO = _FakeYOLO
    entry = get_ultralytics_entry("anzhc_face_seg")
    service = UltralyticsLoaderService(
        folder_paths_module=folder_paths,
        ultralytics_module=ultralytics_module,
        downloader=downloader,
        choice_service=_choice_service(show_downloadable_models=True),
        cache={},
    )

    loaded = service.load(entry.display_name)

    expected_path = models_dir / "ultralytics" / "segm" / "Anzhc Face -seg.pt"
    assert loaded.detector_model.model_path == expected_path
    assert loaded.detector_model.model_name == "segm/Anzhc Face -seg.pt"
    assert loaded.detector_model.supports_segmentation is True
    assert downloader.requests[0].destination_path == expected_path
    assert downloader.requests[0].expected_folder == expected_path.parent
    assert downloader.requests[0].expected_sha256 == entry.artifacts[0].sha256


def test_curated_bbox_model_downloads_to_impact_pack_compatible_folder(
    tmp_path: Path,
) -> None:
    """A curated bbox selection downloads into the conventional bbox folder."""

    models_dir = tmp_path / "models"
    folder_paths = _folder_paths(models_dir)
    downloader = _RecordingDownloader()
    ultralytics_module = ModuleType("ultralytics")
    cast(Any, ultralytics_module).YOLO = _FakeYOLO
    entry = get_ultralytics_entry("bingsu_hand_yolov8n")
    service = UltralyticsLoaderService(
        folder_paths_module=folder_paths,
        ultralytics_module=ultralytics_module,
        downloader=downloader,
        choice_service=_choice_service(show_downloadable_models=True),
        cache={},
    )

    loaded = service.load(entry.display_name)

    expected_path = models_dir / "ultralytics" / "bbox" / "hand_yolov8n.pt"
    assert loaded.detector_model.model_path == expected_path
    assert loaded.detector_model.supports_segmentation is False
    assert downloader.requests[0].destination_path == expected_path
    assert downloader.requests[0].expected_sha256 == entry.artifacts[0].sha256


def test_curated_existing_model_must_match_its_catalog_checksum(
    tmp_path: Path,
) -> None:
    """A pre-existing curated checkpoint cannot bypass checksum verification."""

    models_dir = tmp_path / "models"
    checkpoint = models_dir / "ultralytics" / "segm" / "Anzhc Face -seg.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"wrong checkpoint")
    folder_paths = _folder_paths(models_dir)
    service = UltralyticsLoaderService(
        folder_paths_module=folder_paths,
        choice_service=_choice_service(show_downloadable_models=True),
        cache={},
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        service.load("Anzhc Face -seg (6.52MB)")


def test_catalog_and_local_selection_share_one_loaded_model(tmp_path: Path) -> None:
    """Catalog and conventional-path selections share the loaded model instance."""

    models_dir = tmp_path / "models"
    folder_paths = _folder_paths(models_dir)
    downloader = _RecordingDownloader()
    ultralytics_module = ModuleType("ultralytics")
    yolo_factory = _RecordingYOLOFactory()
    cast(Any, ultralytics_module).YOLO = yolo_factory
    entry = get_ultralytics_entry("anzhc_face_seg")
    cache: dict[UltralyticsModelCacheKey, LoadedUltralyticsDetector] = {}
    service = UltralyticsLoaderService(
        folder_paths_module=folder_paths,
        ultralytics_module=ultralytics_module,
        downloader=downloader,
        choice_service=_choice_service(show_downloadable_models=True),
        cache=cache,
    )

    catalog_loaded = service.load(entry.display_name)
    local_loaded = service.load("segm/Anzhc Face -seg.pt")

    assert local_loaded is catalog_loaded
    assert len(downloader.requests) == 1
    assert len(yolo_factory.paths) == 1
    assert len(cache) == 1


def test_bbox_prefix_marks_model_as_bbox_only(tmp_path: Path) -> None:
    """BBox-prefixed models do not claim segmentation support."""

    models_dir = tmp_path / "models"
    (models_dir / "ultralytics" / "bbox").mkdir(parents=True)
    (models_dir / "ultralytics" / "bbox" / "face.pt").write_bytes(b"")
    ultralytics_module = ModuleType("ultralytics")
    cast(Any, ultralytics_module).YOLO = _FakeYOLO

    service = UltralyticsLoaderService(
        folder_paths_module=_folder_paths(models_dir),
        ultralytics_module=ultralytics_module,
    )

    loaded = service.load("bbox/face.pt")

    assert loaded.detector_model.supports_segmentation is False


def test_loader_uses_process_cache_for_identical_selection(tmp_path: Path) -> None:
    """Identical Ultralytics selections reuse the same loaded detector bundle."""

    models_dir = tmp_path / "models"
    (models_dir / "ultralytics").mkdir(parents=True)
    (models_dir / "ultralytics" / "model.pt").write_bytes(b"")
    ultralytics_module = ModuleType("ultralytics")
    yolo_factory = _RecordingYOLOFactory()
    cast(Any, ultralytics_module).YOLO = yolo_factory
    cache: dict[UltralyticsModelCacheKey, LoadedUltralyticsDetector] = {}
    service = UltralyticsLoaderService(
        folder_paths_module=_folder_paths(models_dir),
        ultralytics_module=ultralytics_module,
        cache=cache,
    )

    first = service.load("model.pt")
    second = service.load("model.pt")

    assert second is first
    assert yolo_factory.paths == [str(models_dir / "ultralytics" / "model.pt")]
    assert len(cache) == 1


def test_loader_cache_separates_prefixed_selections(tmp_path: Path) -> None:
    """BBox and segmentation selections stay separate cache entries."""

    models_dir = tmp_path / "models"
    (models_dir / "ultralytics" / "bbox").mkdir(parents=True)
    (models_dir / "ultralytics" / "segm").mkdir(parents=True)
    (models_dir / "ultralytics" / "bbox" / "face.pt").write_bytes(b"")
    (models_dir / "ultralytics" / "segm" / "face.pt").write_bytes(b"")
    ultralytics_module = ModuleType("ultralytics")
    yolo_factory = _RecordingYOLOFactory()
    cast(Any, ultralytics_module).YOLO = yolo_factory
    cache: dict[UltralyticsModelCacheKey, LoadedUltralyticsDetector] = {}
    service = UltralyticsLoaderService(
        folder_paths_module=_folder_paths(models_dir),
        ultralytics_module=ultralytics_module,
        cache=cache,
    )

    first = service.load("bbox/face.pt")
    second = service.load("segm/face.pt")

    assert second is not first
    assert first.detector_model.supports_segmentation is False
    assert second.detector_model.supports_segmentation is True
    assert yolo_factory.paths == [
        str(models_dir / "ultralytics" / "bbox" / "face.pt"),
        str(models_dir / "ultralytics" / "segm" / "face.pt"),
    ]
    assert len(cache) == 2


def test_loader_does_not_cache_failed_yolo_construction(tmp_path: Path) -> None:
    """A failed YOLO construction leaves the cache empty for retry."""

    models_dir = tmp_path / "models"
    (models_dir / "ultralytics").mkdir(parents=True)
    (models_dir / "ultralytics" / "model.pt").write_bytes(b"")
    ultralytics_module = ModuleType("ultralytics")
    yolo_factory = _RecordingYOLOFactory(fail_once=True)
    cast(Any, ultralytics_module).YOLO = yolo_factory
    cache: dict[UltralyticsModelCacheKey, LoadedUltralyticsDetector] = {}
    service = UltralyticsLoaderService(
        folder_paths_module=_folder_paths(models_dir),
        ultralytics_module=ultralytics_module,
        cache=cache,
    )

    with pytest.raises(RuntimeError, match="could not be loaded"):
        service.load("model.pt")

    loaded = service.load("model.pt")

    assert isinstance(loaded, LoadedUltralyticsDetector)
    assert len(yolo_factory.paths) == 2
    assert len(cache) == 1


class _FakeYOLO:
    """Small fake for Ultralytics YOLO construction."""

    task = "detect"
    names = {0: "face"}

    def __init__(self, path: str) -> None:
        """Record the model path used for loading."""

        self.path = path


class _RecordingYOLOFactory:
    """Callable fake YOLO constructor with call recording."""

    def __init__(self, fail_once: bool = False) -> None:
        """Create a recording YOLO factory."""

        self.paths: list[str] = []
        self.fail_once = fail_once

    def __call__(self, path: str) -> _FakeYOLO:
        """Record the requested model path and optionally fail once."""

        self.paths.append(path)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("YOLO failed")
        return _FakeYOLO(path)


class _RecordingDownloader(ModelDownloader):
    """Download boundary double that records verified catalog requests."""

    def __init__(self) -> None:
        """Initialize the recorded request collection."""

        self.requests: list[DownloadRequest] = []

    def download(
        self,
        request: DownloadRequest,
        progress: ProgressReporter | None = None,
    ) -> DownloadResult:
        """Materialize a placeholder checkpoint at the requested destination."""

        del progress
        self.requests.append(request)
        request.destination_path.parent.mkdir(parents=True, exist_ok=True)
        request.destination_path.write_bytes(b"checkpoint")
        return DownloadResult(
            path=request.destination_path,
            bytes_downloaded=len(b"checkpoint"),
            skipped_existing=False,
        )


class _FakeSettingsRepository:
    """Settings boundary double for Ultralytics dropdown tests."""

    def __init__(self, show_downloadable_models: bool) -> None:
        """Store the configured dropdown visibility preference."""

        self._settings = SimpleSyrupSettings(
            show_downloadable_models=show_downloadable_models
        )

    def load(self) -> SimpleSyrupSettings:
        """Return the configured settings value."""

        return self._settings


def _choice_service(
    *,
    show_downloadable_models: bool,
) -> ModelChoiceService:
    """Build an Ultralytics choice service with deterministic settings."""

    return ModelChoiceService(_FakeSettingsRepository(show_downloadable_models))


def _folder_paths(models_dir: Path) -> ModuleType:
    """Build a minimal fake ComfyUI folder_paths module."""

    module = ModuleType("folder_paths")
    module_any = cast(Any, module)
    module_any.models_dir = str(models_dir)
    module_any.folder_names_and_paths = {}

    def add_model_folder_path(folder_name: str, path: str) -> None:
        module_any.folder_names_and_paths[folder_name] = (
            [path],
            {".pt", ".pth", ".safetensors"},
        )

    def get_filename_list(folder_name: str) -> list[str]:
        paths = module_any.folder_names_and_paths.get(folder_name, ([], set()))[0]
        names: list[str] = []
        for folder in paths:
            root = Path(str(folder))
            if root.is_dir():
                names.extend(path.name for path in root.iterdir() if path.is_file())
        return names

    module_any.add_model_folder_path = add_model_folder_path
    module_any.get_filename_list = get_filename_list
    return module
