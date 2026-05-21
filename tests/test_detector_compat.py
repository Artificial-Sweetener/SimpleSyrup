# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for detector compatibility facades."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.runtime.detector_compat import BBoxDetectorFacade, SegmDetectorFacade
from simple_syrup.runtime.ultralytics_loader import UltralyticsDetectorModel


def test_bbox_facade_accepts_detector_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BBox facade accepts the expected detector arguments."""

    _patch_service(monkeypatch, prefer_segmentation_expected=False)
    facade = BBoxDetectorFacade(_model(supports_segmentation=False))

    header, segments = cast(
        tuple[object, list[Segment]],
        facade.detect(torch.zeros((1, 8, 8, 3)), 0.5, 1, 2.0, 3, None),
    )

    assert header == (8, 8)
    assert segments[0].label == "face"


def test_segmentation_facade_has_bbox_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segmentation facade exposes its paired bbox detector."""

    _patch_service(monkeypatch, prefer_segmentation_expected=True)
    bbox = BBoxDetectorFacade(_model(supports_segmentation=True))
    facade = SegmDetectorFacade(bbox.detector_model, bbox)

    facade.detect(torch.zeros((1, 8, 8, 3)), 0.5, 1, 2.0)

    assert facade.bbox_detector is bbox


def test_segmentation_facade_falls_back_for_bbox_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bbox-only native model remains safe on the segmentation facade."""

    _patch_service(monkeypatch, prefer_segmentation_expected=False)
    bbox = BBoxDetectorFacade(_model(supports_segmentation=False))
    facade = SegmDetectorFacade(bbox.detector_model, bbox)

    header, segments = cast(
        tuple[object, list[Segment]],
        facade.detect(torch.zeros((1, 8, 8, 3)), 0.5, 0, 1.0),
    )

    assert header == (8, 8)
    assert segments[0].label == "face"


class _FakeService:
    """Fake native detection service for facade tests."""

    expected_prefer_segmentation: bool = False

    def detect(
        self,
        image: object,
        detector_model: object,
        threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int,
        prefer_segmentation: bool = True,
    ) -> object:
        """Return deterministic SEGS and verify delegation flags."""

        del image, detector_model, threshold, dilation, crop_factor, drop_size
        assert prefer_segmentation is self.expected_prefer_segmentation
        segment = Segment(
            cropped_image=None,
            cropped_mask=torch.ones((4, 4)),
            confidence=1.0,
            crop_region=CropRegion(0, 0, 4, 4),
            bbox=BoundingBox(1, 1, 3, 3),
            label="face",
        )
        return (8, 8), (segment,)


def _patch_service(
    monkeypatch: pytest.MonkeyPatch,
    prefer_segmentation_expected: bool,
) -> None:
    """Patch the lazily imported detection service used by facades."""

    import simple_syrup.services.segs_detection_service as service_module

    _FakeService.expected_prefer_segmentation = prefer_segmentation_expected
    monkeypatch.setattr(service_module, "SegsDetectionService", _FakeService)


def _model(supports_segmentation: bool) -> UltralyticsDetectorModel:
    """Return a native detector model test double."""

    return UltralyticsDetectorModel(
        model_name="model.pt",
        model_path=Path("model.pt"),
        model=object(),
        task="segment" if supports_segmentation else "detect",
        names={0: "face"},
        supports_segmentation=supports_segmentation,
    )
