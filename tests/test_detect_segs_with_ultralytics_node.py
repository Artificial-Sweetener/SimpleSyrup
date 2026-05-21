# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Detect SEGS w/ Ultralytics node contract."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from simple_syrup.domain.segs import (
    SORT_ORDER_OPTIONS,
    BoundingBox,
    CropRegion,
    NativeSegs,
    Segment,
)
from simple_syrup.nodes.detect_segs_with_ultralytics import DetectSEGSWithUltralytics
from simple_syrup.runtime.ultralytics_loader import UltralyticsDetectorModel
from simple_syrup.services.segs_output_service import CombinedSegsResult


def test_detect_segs_with_ultralytics_node_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node exposes the two-output SEGS and mask contract."""

    monkeypatch.setattr(
        DetectSEGSWithUltralytics,
        "service_class",
        _FakeSegsService,
    )

    inputs = DetectSEGSWithUltralytics.INPUT_TYPES()

    assert DetectSEGSWithUltralytics.RETURN_TYPES == ("SEGS", "MASK")
    assert DetectSEGSWithUltralytics.RETURN_NAMES == ("segs", "mask")
    assert DetectSEGSWithUltralytics.OUTPUT_IS_LIST == (True, False)
    assert DetectSEGSWithUltralytics.OUTPUT_TOOLTIPS == (
        "Detected regions as separate or combined SEGS based on combine_segs.",
        "Combined detected area as a standard ComfyUI mask.",
    )
    assert DetectSEGSWithUltralytics.CATEGORY == "SimpleSyrup/Detection"
    assert DetectSEGSWithUltralytics.DESCRIPTION == (
        "Detects regions with an Ultralytics model and returns individual SEGS, "
        "combined SEGS when requested, and a combined mask."
    )
    assert list(inputs["required"]) == [
        "image",
        "detector_model",
        "confidence_threshold",
        "size_threshold",
        "bbox_dilation",
        "sub_dilation",
        "post_dilation",
        "crop_factor",
        "sort_order",
        "combine_segs",
    ]
    assert inputs["required"]["bbox_dilation"][1]["min"] == -512
    assert inputs["required"]["size_threshold"][1]["default"] == 10
    assert inputs["required"]["post_dilation"][1]["default"] == 0
    assert inputs["required"]["crop_factor"][1]["min"] == 0.0
    assert inputs["required"]["sort_order"][0] == SORT_ORDER_OPTIONS
    assert inputs["required"]["sort_order"][1]["default"] == "largest to smallest"
    assert inputs["required"]["combine_segs"][0] == "BOOLEAN"
    assert inputs["required"]["combine_segs"][1]["default"] is False
    assert "bbox_threshold" not in inputs["required"]
    assert "sub_threshold" not in inputs["required"]
    assert "drop_size" not in inputs["required"]
    assert "optional" not in inputs
    assert "hidden" not in inputs
    for input_declaration in inputs["required"].values():
        assert "tooltip" in input_declaration[1]


def test_detector_returns_individual_segs_when_combine_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SEGS output keeps individual regions when combine_segs is false."""

    builder = _FakeCombinedBuilder()
    monkeypatch.setattr(
        DetectSEGSWithUltralytics,
        "service_class",
        _FakeSegsService,
    )
    monkeypatch.setattr(DetectSEGSWithUltralytics, "combined_builder", builder)

    segs, mask = _detect(combine_segs=False)

    segs_list = cast(list[tuple[object, list[Segment]]], segs)
    assert len(segs_list) == 1
    header, segments = segs_list[0]
    assert header == (8, 8)
    assert isinstance(segments, list)
    assert [segment.label for segment in segments] == ["face"]
    assert builder.call_count == 1
    assert torch.equal(cast(torch.Tensor, mask), torch.ones((1, 8, 8)))


def test_detector_returns_unioned_segs_when_combine_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SEGS output uses the combined region when combine_segs is true."""

    builder = _FakeCombinedBuilder()
    monkeypatch.setattr(
        DetectSEGSWithUltralytics,
        "service_class",
        _FakeSegsService,
    )
    monkeypatch.setattr(DetectSEGSWithUltralytics, "combined_builder", builder)

    segs, mask = _detect(combine_segs=True)

    segs_list = cast(list[tuple[object, list[Segment]]], segs)
    header, segments = segs_list[0]
    assert header == (8, 8)
    assert [segment.label for segment in segments] == ["combined"]
    assert builder.call_count == 1
    assert torch.equal(cast(torch.Tensor, mask), torch.ones((1, 8, 8)))


def test_detect_segs_with_ultralytics_node_uses_story_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detector controls execute in the same order the node declares them."""

    monkeypatch.setattr(
        DetectSEGSWithUltralytics,
        "service_class",
        _FakeStoryOrderSegsService,
    )

    DetectSEGSWithUltralytics().detect(
        torch.zeros((1, 8, 8, 3)),
        _model(),
        0.7,
        12,
        -2,
        3,
        -1,
        2.5,
        "largest to smallest",
        False,
    )


def test_detector_node_sorts_segs_before_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detected SEGS are sorted before normal and combined outputs are built."""

    builder = _FakeCombinedBuilder()
    monkeypatch.setattr(
        DetectSEGSWithUltralytics,
        "service_class",
        _FakeUnsortedSegsService,
    )
    monkeypatch.setattr(DetectSEGSWithUltralytics, "combined_builder", builder)

    segs, _mask = DetectSEGSWithUltralytics().detect(
        torch.zeros((1, 8, 8, 3)),
        _model(),
        0.7,
        1,
        0,
        0,
        0,
        1.0,
        "largest to smallest",
        False,
    )

    segs_list = cast(list[tuple[object, list[Segment]]], segs)
    _header, segments = segs_list[0]
    assert [segment.label for segment in segments] == ["large", "small"]
    assert builder.seen_labels == [["large", "small"]]


def test_detector_node_processes_image_batches_with_individual_segs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch image produces one individual SEGS output per image."""

    builder = _FakeBatchCombinedBuilder()
    monkeypatch.setattr(
        DetectSEGSWithUltralytics,
        "service_class",
        _FakeBatchSegsService,
    )
    monkeypatch.setattr(DetectSEGSWithUltralytics, "combined_builder", builder)
    image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
    image[1] = 1.0

    segs, mask = DetectSEGSWithUltralytics().detect(
        image,
        _model(),
        0.5,
        1,
        0,
        0,
        0,
        1.0,
        "largest to smallest",
        False,
    )

    segs_list = cast(list[tuple[object, list[Segment]]], segs)
    assert len(segs_list) == 2
    assert [segments[0].label for _header, segments in segs_list] == [
        "image-0",
        "image-1",
    ]
    assert [header for header, _segments in segs_list] == [(8, 8), (8, 8)]
    assert builder.call_count == 2
    assert cast(torch.Tensor, mask).shape == (2, 8, 8)


def test_detector_node_processes_image_batches_with_unioned_segs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch image produces one combined SEGS output per image."""

    builder = _FakeBatchCombinedBuilder()
    monkeypatch.setattr(
        DetectSEGSWithUltralytics,
        "service_class",
        _FakeBatchSegsService,
    )
    monkeypatch.setattr(DetectSEGSWithUltralytics, "combined_builder", builder)

    segs, mask = DetectSEGSWithUltralytics().detect(
        torch.zeros((2, 8, 8, 3), dtype=torch.float32),
        _model(),
        0.5,
        1,
        0,
        0,
        0,
        1.0,
        "largest to smallest",
        True,
    )

    segs_list = cast(list[tuple[object, list[Segment]]], segs)
    assert [segments[0].label for _header, segments in segs_list] == [
        "combined-image-0",
        "combined-image-1",
    ]
    assert builder.call_count == 2
    assert cast(torch.Tensor, mask).shape == (2, 8, 8)


class _FakeSegsService:
    """Fake SEGS service for node tests."""

    def detect_simple(
        self,
        image: object,
        detector_model: UltralyticsDetectorModel,
        bbox_threshold: float,
        bbox_dilation: int,
        crop_factor: float,
        drop_size: int,
        sub_threshold: float,
        sub_dilation: int,
        post_dilation: int = 0,
    ) -> NativeSegs:
        """Return deterministic native SEGS."""

        del image, detector_model, crop_factor, drop_size
        assert bbox_threshold == 0.5
        assert bbox_dilation == 0
        assert sub_threshold == 0.5
        assert sub_dilation == 2
        assert post_dilation == -1
        return (8, 8), (_segment("face", CropRegion(0, 0, 2, 2)),)


class _FakeUnsortedSegsService:
    """Fake SEGS service that returns segments out of sorted order."""

    def detect_simple(
        self,
        image: object,
        detector_model: UltralyticsDetectorModel,
        bbox_threshold: float,
        bbox_dilation: int,
        crop_factor: float,
        drop_size: int,
        sub_threshold: float,
        sub_dilation: int,
        post_dilation: int = 0,
    ) -> NativeSegs:
        """Return small then large segments while checking threshold forwarding."""

        del image, detector_model, bbox_dilation, crop_factor, drop_size
        del sub_dilation, post_dilation
        assert bbox_threshold == 0.7
        assert sub_threshold == 0.7
        return (
            (8, 8),
            (
                _segment("small", CropRegion(0, 0, 2, 2), confidence=0.9),
                _segment("large", CropRegion(0, 0, 4, 4), confidence=0.5),
            ),
        )


class _FakeBatchSegsService:
    """Fake SEGS service that returns one labeled segment per image call."""

    def __init__(self) -> None:
        """Create an empty image-call counter."""

        self._call_count = 0

    def detect_simple(
        self,
        image: object,
        detector_model: UltralyticsDetectorModel,
        bbox_threshold: float,
        bbox_dilation: int,
        crop_factor: float,
        drop_size: int,
        sub_threshold: float,
        sub_dilation: int,
        post_dilation: int = 0,
    ) -> NativeSegs:
        """Return a label identifying the one-image slice processed."""

        del detector_model, bbox_threshold, bbox_dilation, crop_factor
        del drop_size, sub_threshold, sub_dilation, post_dilation
        assert cast(torch.Tensor, image).shape == (1, 8, 8, 3)
        label = f"image-{self._call_count}"
        self._call_count += 1
        return (8, 8), (_segment(label, CropRegion(0, 0, 2, 2)),)


class _FakeStoryOrderSegsService:
    """Fake SEGS service that verifies public detector input mapping."""

    def detect_simple(
        self,
        image: object,
        detector_model: UltralyticsDetectorModel,
        bbox_threshold: float,
        bbox_dilation: int,
        crop_factor: float,
        drop_size: int,
        sub_threshold: float,
        sub_dilation: int,
        post_dilation: int = 0,
    ) -> NativeSegs:
        """Check every public input is forwarded to the intended service field."""

        del image, detector_model
        assert bbox_threshold == 0.7
        assert sub_threshold == 0.7
        assert drop_size == 12
        assert bbox_dilation == -2
        assert sub_dilation == 3
        assert post_dilation == -1
        assert crop_factor == 2.5
        return (8, 8), ()


class _FakeCombinedBuilder:
    """Callable fake for combined-output tests."""

    def __init__(self) -> None:
        """Create empty call tracking."""

        self.call_count = 0
        self.seen_labels: list[list[str]] = []

    def __call__(self, image: object, segs: NativeSegs) -> CombinedSegsResult:
        """Return deterministic combined outputs and record the call."""

        del image
        self.call_count += 1
        header, segments = segs
        self.seen_labels.append([segment.label for segment in segments])
        return CombinedSegsResult(
            segs=(header, (_segment("combined", CropRegion(0, 0, 4, 4)),)),
            mask=torch.ones((1, header[0], header[1]), dtype=torch.float32),
        )


class _FakeBatchCombinedBuilder:
    """Callable fake that creates combined labels from source labels."""

    def __init__(self) -> None:
        """Create empty call tracking."""

        self.call_count = 0

    def __call__(self, image: object, segs: NativeSegs) -> CombinedSegsResult:
        """Return a combined segment tied to the source image label."""

        del image
        self.call_count += 1
        header, segments = segs
        label = f"combined-{segments[0].label}" if segments else "combined"
        return CombinedSegsResult(
            segs=(header, (_segment(label, CropRegion(0, 0, 4, 4)),)),
            mask=torch.ones((1, header[0], header[1]), dtype=torch.float32),
        )


def _detect(combine_segs: bool) -> tuple[object, object]:
    """Run the detector node with standard single-image settings."""

    return DetectSEGSWithUltralytics().detect(
        torch.zeros((1, 8, 8, 3)),
        _model(),
        0.5,
        1,
        0,
        2,
        -1,
        1.0,
        "largest to smallest",
        combine_segs,
    )


def _segment(
    label: str,
    crop_region: CropRegion,
    confidence: float = 1.0,
) -> Segment:
    """Create a deterministic test segment."""

    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones((crop_region.height, crop_region.width)),
        confidence=confidence,
        crop_region=crop_region,
        bbox=BoundingBox(
            crop_region.left,
            crop_region.top,
            crop_region.right,
            crop_region.bottom,
        ),
        label=label,
    )


def _model() -> UltralyticsDetectorModel:
    """Return a native detector model test double."""

    return UltralyticsDetectorModel(
        model_name="model.pt",
        model_path=Path("model.pt"),
        model=object(),
        task="segment",
        names={0: "face"},
        supports_segmentation=True,
    )
