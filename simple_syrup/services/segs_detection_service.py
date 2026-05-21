# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Service for converting detector model predictions into native SEGS."""

from __future__ import annotations

from typing import Protocol

import torch

from ..domain.segs import NativeSegs, Segment
from ..masking.segs_mask_ops import (
    crop_image,
    crop_mask,
    crop_region_for_bbox,
    dilate_mask,
    rectangular_mask,
    validate_single_image,
)
from ..runtime.ultralytics_detection import (
    UltralyticsDetection,
    run_ultralytics_detection,
)
from ..runtime.ultralytics_loader import UltralyticsDetectorModel
from ..shared.logging import get_logger
from .segs_output_service import coerce_cropped_mask, combined_mask_from_segs

LOGGER = get_logger(__name__)


class DetectionRunner(Protocol):
    """Callable boundary that runs detector inference."""

    def __call__(
        self,
        detector_model: UltralyticsDetectorModel,
        image: torch.Tensor,
        threshold: float,
        prefer_segmentation: bool,
    ) -> tuple[UltralyticsDetection, ...]:
        """Return parsed detections for an image."""


class SegsDetectionService:
    """Build native SEGS from an Ultralytics detector model."""

    def __init__(
        self,
        detection_runner: DetectionRunner = run_ultralytics_detection,
    ) -> None:
        """Create the service with an injectable inference boundary."""

        self._detection_runner = detection_runner

    def detect(
        self,
        image: object,
        detector_model: UltralyticsDetectorModel,
        threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int,
        prefer_segmentation: bool = True,
        labels: str | None = "all",
        post_dilation: int = 0,
    ) -> NativeSegs:
        """Detect image regions and return immutable native SEGS."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if drop_size < 1:
            raise ValueError("drop_size must be at least 1.")

        image_tensor = validate_single_image(image, "SEGS detector")
        height = int(image_tensor.shape[1])
        width = int(image_tensor.shape[2])
        use_segmentation = prefer_segmentation and detector_model.supports_segmentation
        label_filter = _parse_labels(labels)
        detections = self._detection_runner(
            detector_model,
            image_tensor,
            threshold,
            use_segmentation,
        )

        segments: list[Segment] = []
        for detection in detections:
            if detection.confidence < threshold:
                continue
            if detection.bbox.width < drop_size or detection.bbox.height < drop_size:
                continue
            if label_filter is not None and not _label_matches(
                detection.label,
                label_filter,
            ):
                continue
            mask = detection.mask
            if mask is None or not use_segmentation:
                mask = rectangular_mask(height, width, detection.bbox)
            mask = dilate_mask(mask, dilation).to(device=image_tensor.device)
            crop_region = crop_region_for_bbox(
                detection.bbox,
                image_height=height,
                image_width=width,
                crop_factor=crop_factor,
            )
            cropped_segment_mask = crop_mask(mask, crop_region).detach().clone()
            if post_dilation != 0:
                cropped_segment_mask = dilate_mask(cropped_segment_mask, post_dilation)
            segments.append(
                Segment(
                    cropped_image=crop_image(image_tensor, crop_region)
                    .detach()
                    .clone(),
                    cropped_mask=cropped_segment_mask,
                    confidence=detection.confidence,
                    crop_region=crop_region,
                    bbox=detection.bbox,
                    label=detection.label,
                )
            )

        LOGGER.debug(
            "Detected SEGS",
            extra={
                "operation": "detect_segs",
                "model_name": detector_model.model_name,
                "threshold": threshold,
                "crop_factor": crop_factor,
                "segment_count": len(segments),
            },
        )
        return (height, width), tuple(segments)

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
        """Run Simple Detector SEGS-style bbox detection with optional refinement."""

        bbox_segs = self.detect(
            image=image,
            detector_model=detector_model,
            threshold=bbox_threshold,
            dilation=bbox_dilation,
            crop_factor=crop_factor,
            drop_size=drop_size,
            prefer_segmentation=False,
            post_dilation=0,
        )
        if detector_model.supports_segmentation:
            refinement_segs = self.detect(
                image=image,
                detector_model=detector_model,
                threshold=sub_threshold,
                dilation=sub_dilation,
                crop_factor=crop_factor,
                drop_size=drop_size,
                prefer_segmentation=True,
                post_dilation=0,
            )
            bbox_segs = _intersect_segs_with_combined_mask(bbox_segs, refinement_segs)
        if post_dilation != 0:
            bbox_segs = _dilate_cropped_segs(bbox_segs, post_dilation)
        return bbox_segs


def _parse_labels(labels: str | None) -> set[str] | None:
    """Parse an Impact-style comma-separated label allowlist."""

    if labels is None or labels.strip() == "":
        return None
    parsed = {label.strip() for label in labels.split(",") if label.strip()}
    if not parsed or "all" in parsed:
        return None
    return parsed


def _label_matches(label: str, labels: set[str]) -> bool:
    """Return whether a segment label passes Impact-style label grouping."""

    if label in labels:
        return True
    if "eyes" in labels and label in {"left_eye", "right_eye"}:
        return True
    if "eyebrows" in labels and label in {"left_eyebrow", "right_eyebrow"}:
        return True
    return "pupils" in labels and label in {"left_pupil", "right_pupil"}


def _intersect_segs_with_combined_mask(
    segs: NativeSegs,
    mask_segs: NativeSegs,
) -> NativeSegs:
    """Apply the combined masks from one SEGS payload to another."""

    header, segments = segs
    combined_mask = combined_mask_from_segs(mask_segs)
    refined_segments: list[Segment] = []
    for segment in segments:
        cropped_mask = coerce_cropped_mask(segment)
        refinement_mask = crop_mask(combined_mask, segment.crop_region)
        refined_segments.append(
            Segment(
                cropped_image=segment.cropped_image,
                cropped_mask=torch.minimum(
                    cropped_mask.float(),
                    refinement_mask.float(),
                ),
                confidence=segment.confidence,
                crop_region=segment.crop_region,
                bbox=segment.bbox,
                label=segment.label,
                control_net_wrapper=segment.control_net_wrapper,
            )
        )
    return header, tuple(refined_segments)


def _dilate_cropped_segs(segs: NativeSegs, dilation: int) -> NativeSegs:
    """Apply signed morphology to each cropped segment mask."""

    header, segments = segs
    return header, tuple(
        Segment(
            cropped_image=segment.cropped_image,
            cropped_mask=dilate_mask(coerce_cropped_mask(segment), dilation),
            confidence=segment.confidence,
            crop_region=segment.crop_region,
            bbox=segment.bbox,
            label=segment.label,
            control_net_wrapper=segment.control_net_wrapper,
        )
        for segment in segments
    )
