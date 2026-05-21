# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Prompt SEGS w/ SAM application service."""

from __future__ import annotations

from typing import Protocol, cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, NativeSegs
from simple_syrup.masking.prompt_segs_with_sam_service import (
    PromptSegsRuntime,
    PromptSEGSWithSAMService,
)
from simple_syrup.runtime.text_box_detector import TextBoxDetection
from test_helpers import make_image_tensor


class RecordingDetector:
    """Text box detector double that records prompts."""

    def __init__(self, detections: dict[str, tuple[TextBoxDetection, ...]]) -> None:
        """Create a detector double."""

        self.detections = detections
        self.calls: list[tuple[str, float]] = []

    def detect(
        self,
        grounding_dino_model: object,
        image: torch.Tensor,
        prompt: str,
        threshold: float,
        execution_device: str,
    ) -> tuple[TextBoxDetection, ...]:
        """Record the prompt and return configured detections."""

        self.calls.append((prompt, threshold))
        return self.detections.get(prompt, ())


class RecordingSegmenter:
    """SAM segmenter double that returns configured masks."""

    def __init__(self, masks: dict[str, list[torch.Tensor]]) -> None:
        """Create a segmenter double."""

        self.masks = masks
        self.calls: list[torch.Tensor] = []
        self.call_index = 0

    def segment_boxes(
        self,
        sam_model: object,
        image: torch.Tensor,
        boxes: torch.Tensor,
        threshold: float,
        execution_device: str,
    ) -> tuple[torch.Tensor, ...]:
        """Record boxes and return the next mask group."""

        self.calls.append(boxes.clone())
        key = str(self.call_index)
        self.call_index += 1
        return tuple(self.masks.get(key, []))


class _HasMaxSizePixels(Protocol):
    """Expose the refinement max-size setting used by the test double."""

    max_size_pixels: int


class RecordingRefiner:
    """Mask detail refiner double."""

    def __init__(self) -> None:
        """Create a refiner double."""

        self.calls: list[int] = []

    def refine(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        settings: object,
        vitmatte_model: object | None,
    ) -> torch.Tensor:
        """Record max size and return the mask."""

        self.calls.append(int(cast(_HasMaxSizePixels, settings).max_size_pixels))
        return mask


def test_service_requires_positive_prompt() -> None:
    """Blank positive prompts are rejected before runtime work."""

    service = _service(RecordingDetector({}), RecordingSegmenter({}))

    with pytest.raises(ValueError, match="positive_prompt is required"):
        _prompt(service, positive_prompt=" ")


def test_service_rejects_invalid_confidence_threshold() -> None:
    """Confidence threshold must be probability-like."""

    service = _service(RecordingDetector({}), RecordingSegmenter({}))

    with pytest.raises(ValueError, match="confidence_threshold"):
        _prompt(service, confidence_threshold=1.5)


def test_service_rejects_invalid_size_threshold() -> None:
    """Size threshold must be positive."""

    service = _service(RecordingDetector({}), RecordingSegmenter({}))

    with pytest.raises(ValueError, match="size_threshold"):
        _prompt(service, size_threshold=0)


def test_service_rejects_invalid_crop_factor() -> None:
    """Crop factor must be compatible with SEGS crops."""

    service = _service(RecordingDetector({}), RecordingSegmenter({}))

    for crop_factor in (0.0, 0.5):
        with pytest.raises(ValueError, match="crop_factor"):
            _prompt(service, crop_factor=crop_factor)


def test_service_skips_blank_negative_prompt() -> None:
    """Blank negative prompt does not run a second detection."""

    detector = RecordingDetector({"face": (_detection(0, 0, 2, 2, 0.8),)})
    service = _service(detector, RecordingSegmenter({"0": [_mask_square(0, 0, 2, 2)]}))

    _prompt(service, positive_prompt=" face ", negative_prompt=" ")

    assert detector.calls == [("face", 0.3)]


def test_service_creates_one_seg_per_positive_box() -> None:
    """Multiple positive detections become multiple SEGS."""

    detector = RecordingDetector(
        {
            "face": (
                _detection(0, 0, 2, 2, 0.8),
                _detection(2, 2, 4, 4, 0.7),
            )
        }
    )
    segmenter = RecordingSegmenter(
        {"0": [_mask_square(0, 0, 2, 2), _mask_square(2, 2, 4, 4)]}
    )
    service = _service(detector, segmenter)

    segs = _prompt(service, positive_prompt="face", size_threshold=1)

    assert [segment.confidence for segment in segs[1]] == [0.8, 0.7]
    assert [segment.label for segment in segs[1]] == ["face", "face"]


def test_service_subtracts_negative_prompt_from_positive_segs() -> None:
    """Negative prompt masks are removed from each positive SEG."""

    detector = RecordingDetector(
        {
            "person": (_detection(0, 0, 4, 4, 0.9),),
            "hand": (_detection(0, 0, 2, 2, 0.9),),
        }
    )
    segmenter = RecordingSegmenter(
        {
            "0": [torch.ones((4, 4), dtype=torch.float32)],
            "1": [_mask_square(0, 0, 2, 2)],
        }
    )
    service = _service(detector, segmenter)

    segs = _prompt(
        service,
        positive_prompt="person",
        negative_prompt="hand",
        size_threshold=1,
    )

    assert len(segs[1]) == 1
    assert torch.count_nonzero(cast(torch.Tensor, segs[1][0].cropped_mask)) == 12


def test_service_discards_positive_segs_emptied_by_negative_prompt() -> None:
    """Positive masks fully removed by the negative prompt are discarded."""

    detector = RecordingDetector(
        {
            "person": (_detection(0, 0, 2, 2, 0.9),),
            "hand": (_detection(0, 0, 2, 2, 0.9),),
        }
    )
    segmenter = RecordingSegmenter(
        {"0": [_mask_square(0, 0, 2, 2)], "1": [_mask_square(0, 0, 2, 2)]}
    )
    service = _service(detector, segmenter)

    segs = _prompt(service, positive_prompt="person", negative_prompt="hand")

    assert segs[1] == ()


def test_service_discards_small_final_segs() -> None:
    """Size threshold filters the final derived bbox."""

    detector = RecordingDetector({"face": (_detection(0, 0, 1, 1, 0.8),)})
    service = _service(
        detector,
        RecordingSegmenter({"0": [_mask_square(0, 0, 1, 1)]}),
    )

    segs = _prompt(service, positive_prompt="face", size_threshold=2)

    assert segs[1] == ()


def test_service_expands_bboxes_before_sam() -> None:
    """bbox_dilation expands prompt boxes before segmentation."""

    detector = RecordingDetector({"face": (_detection(1, 1, 3, 3, 0.8),)})
    segmenter = RecordingSegmenter({"0": [_mask_square(1, 1, 3, 3)]})
    service = _service(detector, segmenter)

    _prompt(service, positive_prompt="face", bbox_dilation=1, size_threshold=1)

    assert torch.equal(
        segmenter.calls[0],
        torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
    )


def test_service_applies_mask_dilation() -> None:
    """mask_dilation morphs the final per-SEG mask."""

    detector = RecordingDetector({"face": (_detection(1, 1, 2, 2, 0.8),)})
    service = _service(
        detector,
        RecordingSegmenter({"0": [_mask_square(1, 1, 2, 2)]}),
    )

    segs = _prompt(
        service,
        positive_prompt="face",
        size_threshold=1,
        mask_dilation=1,
        crop_factor=1.0,
    )

    assert segs[1][0].bbox == BoundingBox(0, 0, 3, 3)
    assert torch.count_nonzero(cast(torch.Tensor, segs[1][0].cropped_mask)) == 9


def test_service_refine_mask_false_skips_external_refiner() -> None:
    """Disabled refinement skips the ViTMatte refiner."""

    refiner = RecordingRefiner()
    detector = RecordingDetector({"face": (_detection(0, 0, 2, 2, 0.8),)})
    service = _service(
        detector,
        RecordingSegmenter({"0": [_mask_square(0, 0, 2, 2)]}),
        refiner,
    )

    _prompt(service, positive_prompt="face", refine_mask=False)

    assert refiner.calls == []


def test_service_vitmatte_refinement_receives_max_size() -> None:
    """mask_refinement_max_size reaches the external refiner settings."""

    refiner = RecordingRefiner()
    detector = RecordingDetector({"face": (_detection(0, 0, 2, 2, 0.8),)})
    service = _service(
        detector,
        RecordingSegmenter({"0": [_mask_square(0, 0, 2, 2)]}),
        refiner,
    )

    _prompt(
        service,
        positive_prompt="face",
        detail_method="VITMatte",
        refine_mask=True,
        mask_refinement_max_size=512,
    )

    assert refiner.calls == [512]


def test_service_sorts_final_segs() -> None:
    """Final SEGS use the requested shared sorting policy."""

    detector = RecordingDetector(
        {
            "face": (
                _detection(0, 0, 1, 1, 0.5),
                _detection(2, 0, 4, 2, 0.9),
            )
        }
    )
    segmenter = RecordingSegmenter(
        {"0": [_mask_square(0, 0, 1, 1), _mask_square(2, 0, 4, 2)]}
    )
    service = _service(detector, segmenter)

    segs = _prompt(
        service,
        positive_prompt="face",
        size_threshold=1,
        sort_order="highest confidence first",
    )

    assert [segment.confidence for segment in segs[1]] == [0.9, 0.5]


def _service(
    detector: RecordingDetector,
    segmenter: RecordingSegmenter,
    refiner: RecordingRefiner | None = None,
) -> PromptSEGSWithSAMService:
    """Create a service from test doubles."""

    return PromptSEGSWithSAMService(
        runtime=PromptSegsRuntime(detector=detector, segmenter=segmenter),
        vitmatte_refiner=refiner,
    )


def _prompt(
    service: PromptSEGSWithSAMService,
    *,
    positive_prompt: str = "face",
    negative_prompt: str = "",
    confidence_threshold: float = 0.3,
    size_threshold: int = 1,
    bbox_dilation: int = 0,
    mask_dilation: int = 0,
    detail_method: str = "GuidedFilter",
    detail_erode: int = 0,
    detail_dilate: int = 0,
    black_point: float = 0.0,
    white_point: float = 1.0,
    refine_mask: bool = False,
    mask_refinement_max_size: int = 2048,
    execution_device: str = "cpu",
    crop_factor: float = 1.0,
    sort_order: str = "largest to smallest",
) -> NativeSegs:
    """Call the service with common test options."""

    return service.prompt(
        image=make_image_tensor(batch_size=1, height=4, width=4),
        sam_model={"sam": "model"},
        grounding_dino_model={"dino": "model"},
        vitmatte_model={"vitmatte": "model"},
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        confidence_threshold=confidence_threshold,
        size_threshold=size_threshold,
        bbox_dilation=bbox_dilation,
        mask_dilation=mask_dilation,
        detail_method=detail_method,
        detail_erode=detail_erode,
        detail_dilate=detail_dilate,
        black_point=black_point,
        white_point=white_point,
        refine_mask=refine_mask,
        mask_refinement_max_size=mask_refinement_max_size,
        execution_device=execution_device,
        crop_factor=crop_factor,
        sort_order=sort_order,
    )


def _detection(
    left: int,
    top: int,
    right: int,
    bottom: int,
    confidence: float,
) -> TextBoxDetection:
    """Create a text detection for tests."""

    return TextBoxDetection(
        bbox=BoundingBox(left, top, right, bottom),
        confidence=confidence,
    )


def _mask_square(left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    """Create a 4x4 mask with one filled square."""

    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[top:bottom, left:right] = 1.0
    return mask
