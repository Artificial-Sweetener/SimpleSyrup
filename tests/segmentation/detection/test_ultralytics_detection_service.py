# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for converting Ultralytics detections into SEGS."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion
from simple_syrup.runtime.ultralytics_detection import (
    UltralyticsDetection,
    parse_ultralytics_result,
)
from simple_syrup.runtime.ultralytics_model_adapter import UltralyticsDetectorModel
from simple_syrup.services.segs_detection_service import (
    DetectionRunner,
    SegsDetectionService,
)


def test_bbox_prediction_creates_rectangular_mask_segs() -> None:
    """BBox detections produce rectangular cropped masks."""

    service = SegsDetectionService(detection_runner=_runner(_bbox_detection()))

    header, segments = service.detect(_image(), _model(False), 0.5, 0, 1.0, 1)

    assert header == (8, 8)
    assert len(segments) == 1
    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 9.0
    assert segments[0].crop_region == (2, 2, 5, 5)
    assert torch.all(cast(torch.Tensor, segments[0].cropped_mask) == 1.0)


def test_segmentation_prediction_uses_mask_shape() -> None:
    """Segmentation detections preserve the predicted mask values."""

    mask = torch.zeros((8, 8), dtype=torch.float32)
    mask[1:6, 1:6] = 1.0
    detection = UltralyticsDetection(BoundingBox(2, 2, 5, 5), 0.9, "face", mask)
    service = SegsDetectionService(detection_runner=_runner(detection))

    _header, segments = service.detect(_image(), _model(True), 0.5, 0, 1.0, 1)

    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 9.0


def test_threshold_filters_low_confidence_detections() -> None:
    """Low-confidence detections are dropped."""

    detection = UltralyticsDetection(BoundingBox(2, 2, 5, 5), 0.25, "face", None)
    service = SegsDetectionService(detection_runner=_runner(detection))

    _header, segments = service.detect(_image(), _model(False), 0.5, 0, 1.0, 1)

    assert segments == ()


def test_drop_size_filters_tiny_detections() -> None:
    """Detections smaller than drop_size are dropped."""

    detection = UltralyticsDetection(BoundingBox(2, 2, 4, 4), 0.9, "face", None)
    service = SegsDetectionService(detection_runner=_runner(detection))

    _header, segments = service.detect(_image(), _model(False), 0.5, 0, 1.0, 3)

    assert segments == ()


def test_crop_factor_expands_region() -> None:
    """Crop factor expands around the bbox without leaving image bounds."""

    service = SegsDetectionService(detection_runner=_runner(_bbox_detection()))

    _header, segments = service.detect(_image(), _model(False), 0.5, 0, 3.0, 1)

    assert segments[0].crop_region == (0, 0, 8, 8)
    cropped_mask = cast(torch.Tensor, segments[0].cropped_mask)
    assert cropped_mask.sum().item() == 9.0
    assert torch.count_nonzero(cropped_mask == 0.0).item() > 0


def test_crop_factor_shifts_region_at_image_edges() -> None:
    """Edge crops preserve requested context by shifting back into the image."""

    detection = UltralyticsDetection(BoundingBox(0, 2, 2, 4), 0.9, "face", None)
    service = SegsDetectionService(detection_runner=_runner(detection))

    _header, segments = service.detect(_image(), _model(False), 0.5, 0, 3.0, 1)

    assert segments[0].crop_region == CropRegion(0, 0, 6, 6)
    assert cast(torch.Tensor, segments[0].cropped_mask).shape == (6, 6)


def test_crop_factor_zero_uses_full_image_region() -> None:
    """A zero crop factor selects the full image as the SEG crop."""

    service = SegsDetectionService(detection_runner=_runner(_bbox_detection()))

    _header, segments = service.detect(_image(), _model(False), 0.5, 0, 0.0, 1)

    assert segments[0].crop_region == (0, 0, 8, 8)
    assert cast(torch.Tensor, segments[0].cropped_image).shape == (1, 8, 8, 3)
    assert cast(torch.Tensor, segments[0].cropped_mask).shape == (8, 8)


def test_fractional_crop_factor_below_one_still_fails() -> None:
    """Crop factor accepts zero or values at least one, but not in between."""

    service = SegsDetectionService(detection_runner=_runner(_bbox_detection()))

    with pytest.raises(ValueError, match="crop_factor"):
        service.detect(_image(), _model(False), 0.5, 0, 0.5, 1)


def test_dilation_factor_uses_kernel_size() -> None:
    """Positive dilation uses the factor as the morphology kernel size."""

    detection = UltralyticsDetection(BoundingBox(2, 2, 4, 4), 0.9, "face", None)
    service = SegsDetectionService(detection_runner=_runner(detection))

    _header, segments = service.detect(_image(), _model(False), 0.5, 2, 3.0, 1)

    full_mask = torch.zeros((8, 8), dtype=torch.float32)
    region = segments[0].crop_region
    full_mask[region.top : region.bottom, region.left : region.right] = cast(
        torch.Tensor, segments[0].cropped_mask
    )

    assert torch.equal(
        full_mask,
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
    )


def test_negative_dilation_factor_uses_kernel_size() -> None:
    """Negative dilation erodes with the factor as the morphology kernel size."""

    detection = UltralyticsDetection(BoundingBox(1, 1, 7, 7), 0.9, "face", None)
    service = SegsDetectionService(detection_runner=_runner(detection))

    _header, segments = service.detect(_image(), _model(False), 0.5, -3, 1.0, 1)

    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 16.0


def test_labels_filter_segments() -> None:
    """Comma-separated labels keep only matching segments."""

    face = UltralyticsDetection(BoundingBox(1, 1, 4, 4), 0.9, "face", None)
    hand = UltralyticsDetection(BoundingBox(4, 4, 7, 7), 0.9, "hand", None)
    service = SegsDetectionService(detection_runner=_runner(face, hand))

    _header, segments = service.detect(
        _image(),
        _model(False),
        0.5,
        0,
        1.0,
        1,
        labels="face",
    )

    assert [segment.label for segment in segments] == ["face"]


def test_label_groups_match_impact_aliases() -> None:
    """Impact-style label groups are supported without importing Impact."""

    eye = UltralyticsDetection(BoundingBox(1, 1, 4, 4), 0.9, "left_eye", None)
    hand = UltralyticsDetection(BoundingBox(4, 4, 7, 7), 0.9, "hand", None)
    service = SegsDetectionService(detection_runner=_runner(eye, hand))

    _header, segments = service.detect(
        _image(),
        _model(False),
        0.5,
        0,
        1.0,
        1,
        labels="eyes",
    )

    assert [segment.label for segment in segments] == ["left_eye"]


def test_post_dilation_changes_cropped_mask_after_crop() -> None:
    """Post-dilation applies to the cropped SEGS mask."""

    detection = UltralyticsDetection(BoundingBox(2, 2, 5, 5), 0.9, "face", None)
    service = SegsDetectionService(detection_runner=_runner(detection))

    _header, segments = service.detect(
        _image(),
        _model(False),
        0.5,
        0,
        3.0,
        1,
        post_dilation=-3,
    )

    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 1.0


def test_ultralytics_segmentation_masks_drop_aspect_padding_before_resize() -> None:
    """Segmentation masks remove model-shape padding before image normalization."""

    mask = torch.zeros((1, 6, 5), dtype=torch.float32)
    mask[:, 1:5, :] = 1.0
    result = _UltralyticsResult(
        boxes=_Boxes(
            xyxy=torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([0.0]),
        ),
        masks=_Masks(mask),
    )

    detections = parse_ultralytics_result(
        result,
        image_height=4,
        image_width=4,
        class_names={0: "face"},
        prefer_segmentation=True,
    )

    assert len(detections) == 1
    assert detections[0].mask is not None
    assert torch.all(detections[0].mask == 1.0)


def test_simple_detector_refines_bbox_segs_with_segmentation_mask() -> None:
    """Simple detection intersects bbox SEGS with segmentation sub-detections."""

    segmentation_mask = torch.zeros((8, 8), dtype=torch.float32)
    segmentation_mask[3:5, 3:5] = 1.0
    runner = _SequentialRunner(
        (UltralyticsDetection(BoundingBox(2, 2, 6, 6), 0.9, "face", None),),
        (
            UltralyticsDetection(
                BoundingBox(2, 2, 6, 6),
                0.8,
                "face",
                segmentation_mask,
            ),
        ),
    )
    service = SegsDetectionService(detection_runner=runner)

    _header, segments = service.detect_simple(
        _image(),
        _model(True),
        bbox_threshold=0.5,
        bbox_dilation=0,
        crop_factor=1.0,
        drop_size=1,
        sub_threshold=0.6,
        sub_dilation=0,
    )

    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 4.0
    assert runner.calls == [(0.5, False), (0.6, True)]


def test_simple_detector_bbox_only_model_skips_segmentation_refinement() -> None:
    """BBox-only models use the bbox path without a second sub-detection call."""

    runner = _SequentialRunner(
        (UltralyticsDetection(BoundingBox(2, 2, 5, 5), 0.9, "face", None),)
    )
    service = SegsDetectionService(detection_runner=runner)

    _header, segments = service.detect_simple(
        _image(),
        _model(False),
        bbox_threshold=0.5,
        bbox_dilation=0,
        crop_factor=1.0,
        drop_size=1,
        sub_threshold=0.6,
        sub_dilation=0,
    )

    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 9.0
    assert runner.calls == [(0.5, False)]


def test_empty_detections_return_empty_segs() -> None:
    """No detections returns an empty SEGS payload."""

    service = SegsDetectionService(detection_runner=_runner())

    assert service.detect(_image(), _model(False), 0.5, 0, 1.0, 1) == ((8, 8), ())


def test_batch_image_input_fails_clearly() -> None:
    """The first version rejects batched images."""

    service = SegsDetectionService(detection_runner=_runner())

    with pytest.raises(ValueError, match="supports one image at a time"):
        service.detect(torch.zeros((2, 8, 8, 3)), _model(False), 0.5, 0, 1.0, 1)


def _bbox_detection() -> UltralyticsDetection:
    """Return a reusable bbox detection."""

    return UltralyticsDetection(BoundingBox(2, 2, 5, 5), 0.9, "face", None)


def _runner(
    *detections: UltralyticsDetection,
) -> DetectionRunner:
    """Return a detection runner fake."""

    def run(
        detector_model: UltralyticsDetectorModel,
        image: torch.Tensor,
        threshold: float,
        prefer_segmentation: bool,
    ) -> tuple[UltralyticsDetection, ...]:
        del detector_model, image, threshold, prefer_segmentation
        return tuple(detections)

    return run


class _SequentialRunner:
    """Detection runner fake that returns a different payload per call."""

    def __init__(self, *responses: tuple[UltralyticsDetection, ...]) -> None:
        """Store ordered fake detection responses."""

        self._responses = list(responses)
        self.calls: list[tuple[float, bool]] = []

    def __call__(
        self,
        detector_model: UltralyticsDetectorModel,
        image: torch.Tensor,
        threshold: float,
        prefer_segmentation: bool,
    ) -> tuple[UltralyticsDetection, ...]:
        """Return the next response and record call parameters."""

        del detector_model, image
        self.calls.append((threshold, prefer_segmentation))
        return self._responses.pop(0)


class _Boxes:
    """Fake Ultralytics boxes object for parser tests."""

    def __init__(
        self,
        xyxy: torch.Tensor,
        conf: torch.Tensor,
        cls: torch.Tensor,
    ) -> None:
        """Store tensor-like Ultralytics box fields."""

        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class _Masks:
    """Fake Ultralytics masks object for parser tests."""

    def __init__(self, data: torch.Tensor) -> None:
        """Store tensor-like Ultralytics mask data."""

        self.data = data


class _UltralyticsResult:
    """Fake Ultralytics result object for parser tests."""

    def __init__(self, boxes: _Boxes, masks: _Masks) -> None:
        """Store result-like fields."""

        self.boxes = boxes
        self.masks = masks
        self.names = {0: "face"}


def _image() -> torch.Tensor:
    """Return a small single-image tensor."""

    return torch.zeros((1, 8, 8, 3), dtype=torch.float32)


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
