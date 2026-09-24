# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for detector compatibility facades."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.runtime.ultralytics_model_adapter import UltralyticsDetectorModel
from simple_syrup.services.detector_compat import BBoxDetectorFacade, SegmDetectorFacade


def test_bbox_facade_accepts_detector_signature() -> None:
    """BBox facade accepts the expected detector arguments."""

    service = _FakeService(expected_prefer_segmentation=False)
    facade = BBoxDetectorFacade(_model(supports_segmentation=False), cast(Any, service))

    header, segments = cast(
        tuple[object, list[Segment]],
        facade.detect(torch.zeros((1, 8, 8, 3)), 0.5, 1, 2.0, 3, None),
    )

    assert header == (8, 8)
    assert segments[0].label == "face"


def test_segmentation_facade_has_bbox_detector() -> None:
    """Segmentation facade exposes its paired bbox detector."""

    service = _FakeService(expected_prefer_segmentation=True)
    bbox = BBoxDetectorFacade(_model(supports_segmentation=True), cast(Any, service))
    facade = SegmDetectorFacade(bbox.detector_model, bbox, cast(Any, service))

    facade.detect(torch.zeros((1, 8, 8, 3)), 0.5, 1, 2.0)

    assert facade.bbox_detector is bbox


def test_segmentation_facade_falls_back_for_bbox_model() -> None:
    """A bbox-only native model remains safe on the segmentation facade."""

    service = _FakeService(expected_prefer_segmentation=False)
    bbox = BBoxDetectorFacade(_model(supports_segmentation=False), cast(Any, service))
    facade = SegmDetectorFacade(bbox.detector_model, bbox, cast(Any, service))

    header, segments = cast(
        tuple[object, list[Segment]],
        facade.detect(torch.zeros((1, 8, 8, 3)), 0.5, 0, 1.0),
    )

    assert header == (8, 8)
    assert segments[0].label == "face"


class _FakeService:
    """Fake native detection service for facade tests."""

    def __init__(self, expected_prefer_segmentation: bool) -> None:
        """Store the expected detection mode."""

        self.expected_prefer_segmentation = expected_prefer_segmentation

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
