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

    service = UltralyticsLoaderService(folder_paths_module=_folder_paths(models_dir))

    assert service.model_choices() == ["bbox/face.pt", "root.pt", "segm/person.pt"]


def test_model_choices_returns_sentinel_when_no_models(tmp_path: Path) -> None:
    """An empty model directory returns a clear dropdown sentinel."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    service = UltralyticsLoaderService(folder_paths_module=_folder_paths(models_dir))

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
