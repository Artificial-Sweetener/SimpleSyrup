# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convert unprompted SAM masks into image-associated SEGS."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from time import perf_counter

import torch
import torch.nn.functional as functional

from ..domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from ..masking.segs_mask_ops import validate_single_image
from ..runtime.progress import NullPhaseProgressReporter, PhaseProgressReporter
from ..runtime.sam_automatic_segmenter import (
    AutomaticSAMMask,
    SAMAutomaticSegmenter,
    SAMModelAutomaticSegmenter,
)
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SAMAutoSegsSettings:
    """Validated controls for converting automatic SAM masks to SEGS."""

    segmentation_resolution: int
    minimum_region_area: int


@dataclass(frozen=True)
class _GuideMaskCandidate:
    """Keep a retained SAM mask in compact guide-image coordinates."""

    mask: torch.Tensor
    bbox: BoundingBox
    confidence: float
    label: str | None


class SEGSFromSAMOutputService:
    """Build reusable SEGS from a SAM model's unprompted masks."""

    def __init__(self, segmenter: SAMAutomaticSegmenter | None = None) -> None:
        """Create the service with an injectable automatic segmentation runtime."""

        self._segmenter = segmenter or SAMModelAutomaticSegmenter()

    def build(
        self,
        *,
        image: object,
        sam_model: object,
        segmentation_resolution: int,
        minimum_region_area: int,
        phase_progress: PhaseProgressReporter | None = None,
    ) -> NativeSegs:
        """Return source-sized SEGS from unprompted SAM masks."""

        operation_started_at = perf_counter()
        reporter = phase_progress or NullPhaseProgressReporter()
        source_image = validate_single_image(image, "SEGS from SAM Output")
        settings = _validate_settings(
            segmentation_resolution=segmentation_resolution,
            minimum_region_area=minimum_region_area,
        )
        image_height = int(source_image.shape[1])
        image_width = int(source_image.shape[2])
        reporter.advance("preparing_segmentation_image")
        guide_image = _resize_for_segmentation(
            source_image,
            maximum_long_edge=settings.segmentation_resolution,
        )
        reporter.advance("generating_automatic_masks")
        generated = self._segmenter.segment_all(sam_model, guide_image)
        masks_generated_at = perf_counter()
        reporter.advance("building_segs")
        candidates = _normalize_masks(
            generated,
            source_height=image_height,
            source_width=image_width,
            minimum_region_area=settings.minimum_region_area,
        )
        segments = tuple(
            _build_segment(
                image=source_image,
                candidate=candidate,
                source_height=image_height,
                source_width=image_width,
                confidence=candidate.confidence,
                label=candidate.label or f"segment_{index:03d}",
            )
            for index, candidate in enumerate(candidates, start=1)
        )
        LOGGER.info(
            "Built SEGS from SAM output",
            extra={
                "operation": "segs_from_sam_output",
                "segmentation_resolution": settings.segmentation_resolution,
                "minimum_region_area": settings.minimum_region_area,
                "source_height": image_height,
                "source_width": image_width,
                "generated_mask_count": len(generated),
                "segment_count": len(segments),
                "mask_generation_ms": round(
                    (masks_generated_at - operation_started_at) * 1000.0,
                    2,
                ),
                "segs_construction_ms": round(
                    (perf_counter() - masks_generated_at) * 1000.0,
                    2,
                ),
            },
        )
        return (image_height, image_width), segments


def _validate_settings(
    *,
    segmentation_resolution: int,
    minimum_region_area: int,
) -> SAMAutoSegsSettings:
    """Validate public automatic-SEGS configuration before model execution."""

    if segmentation_resolution < 64 or segmentation_resolution % 64 != 0:
        raise ValueError(
            "segmentation_resolution must be at least 64 and divisible by 64."
        )
    if minimum_region_area < 0:
        raise ValueError("minimum_region_area must be greater than or equal to 0.")
    return SAMAutoSegsSettings(
        segmentation_resolution=segmentation_resolution,
        minimum_region_area=minimum_region_area,
    )


def _resize_for_segmentation(
    image: torch.Tensor,
    *,
    maximum_long_edge: int,
) -> torch.Tensor:
    """Downscale an image to the requested guide resolution without upscaling it."""

    source_height = int(image.shape[1])
    source_width = int(image.shape[2])
    source_long_edge = max(source_height, source_width)
    if source_long_edge <= maximum_long_edge:
        return image
    scale = maximum_long_edge / source_long_edge
    guide_height = max(1, round(source_height * scale))
    guide_width = max(1, round(source_width * scale))
    resized = functional.interpolate(
        image.movedim(-1, 1),
        size=(guide_height, guide_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _normalize_masks(
    masks: tuple[AutomaticSAMMask, ...],
    *,
    source_height: int,
    source_width: int,
    minimum_region_area: int,
) -> tuple[_GuideMaskCandidate, ...]:
    """Filter and deduplicate masks while they remain at guide resolution."""

    normalized: list[_GuideMaskCandidate] = []
    for candidate in masks:
        binary = _binary_guide_mask(candidate.mask)
        bbox = _bbox_from_mask(binary)
        if bbox is None:
            continue
        if (
            _projected_source_area(
                active_pixels=int(binary.sum().item()),
                guide_height=int(binary.shape[0]),
                guide_width=int(binary.shape[1]),
                source_height=source_height,
                source_width=source_width,
            )
            < minimum_region_area
        ):
            continue
        normalized_candidate = _GuideMaskCandidate(
            mask=binary,
            bbox=bbox,
            confidence=candidate.confidence,
            label=candidate.label,
        )
        duplicate_index = _duplicate_mask_index(normalized, normalized_candidate)
        if duplicate_index is None:
            normalized.append(normalized_candidate)
        elif normalized_candidate.confidence > normalized[duplicate_index].confidence:
            normalized[duplicate_index] = normalized_candidate
    return tuple(normalized)


def _binary_guide_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return one validated binary guide-space mask on the CPU."""

    working = mask.detach().cpu().float()
    if working.ndim != 2:
        raise ValueError("SAM automatic segmentation returned a mask that is not HW.")
    if int(working.shape[0]) == 0 or int(working.shape[1]) == 0:
        raise ValueError("SAM automatic segmentation returned an empty mask.")
    return working >= 0.5


def _projected_source_area(
    *,
    active_pixels: int,
    guide_height: int,
    guide_width: int,
    source_height: int,
    source_width: int,
) -> float:
    """Estimate source-pixel area directly from one guide-space mask."""

    return (
        active_pixels * source_height * source_width / float(guide_height * guide_width)
    )


def _duplicate_mask_index(
    candidates: list[_GuideMaskCandidate],
    candidate: _GuideMaskCandidate,
) -> int | None:
    """Return a near-identical guide mask index without broad pairwise scans."""

    for index, existing in enumerate(candidates):
        if (
            tuple(existing.mask.shape) != tuple(candidate.mask.shape)
            or _bbox_iou(existing.bbox, candidate.bbox) < 0.98
        ):
            continue
        union = torch.logical_or(existing.mask, candidate.mask).sum()
        if int(union.item()) == 0:
            continue
        intersection = torch.logical_and(existing.mask, candidate.mask).sum()
        if float(intersection.item()) / float(union.item()) >= 0.98:
            return index
    return None


def _build_segment(
    *,
    image: torch.Tensor,
    candidate: _GuideMaskCandidate,
    source_height: int,
    source_width: int,
    confidence: float,
    label: str,
) -> Segment:
    """Build one crop-local SEG without ever expanding a full source mask."""

    crop_region = _source_crop_region(
        candidate.bbox,
        guide_height=int(candidate.mask.shape[0]),
        guide_width=int(candidate.mask.shape[1]),
        source_height=source_height,
        source_width=source_width,
    )
    guide_crop = candidate.mask[
        candidate.bbox.top : candidate.bbox.bottom,
        candidate.bbox.left : candidate.bbox.right,
    ]
    local_mask = _resize_local_mask(
        guide_crop,
        height=crop_region.height,
        width=crop_region.width,
    )
    return Segment(
        cropped_image=image[
            :,
            crop_region.top : crop_region.bottom,
            crop_region.left : crop_region.right,
            :,
        ]
        .detach()
        .clone(),
        cropped_mask=local_mask.detach().clone(),
        confidence=max(0.0, min(1.0, float(confidence))),
        crop_region=crop_region,
        bbox=BoundingBox(
            crop_region.left,
            crop_region.top,
            crop_region.right,
            crop_region.bottom,
        ),
        label=label,
    )


def _bbox_from_mask(mask: torch.Tensor) -> BoundingBox | None:
    """Return the tight bounding box for one active guide-space mask."""

    y_coords, x_coords = torch.where(mask)
    if y_coords.numel() == 0:
        return None
    return BoundingBox(
        left=int(x_coords.min().item()),
        top=int(y_coords.min().item()),
        right=int(x_coords.max().item()) + 1,
        bottom=int(y_coords.max().item()) + 1,
    )


def _bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Return the intersection-over-union of two rectangular bounds."""

    overlap_width = max(
        0, min(first.right, second.right) - max(first.left, second.left)
    )
    overlap_height = max(
        0, min(first.bottom, second.bottom) - max(first.top, second.top)
    )
    intersection = overlap_width * overlap_height
    union = first.width * first.height + second.width * second.height - intersection
    return 0.0 if union == 0 else intersection / float(union)


def _source_crop_region(
    guide_bbox: BoundingBox,
    *,
    guide_height: int,
    guide_width: int,
    source_height: int,
    source_width: int,
) -> CropRegion:
    """Map a guide-space bounding box to a conservative source-image crop."""

    left = _map_lower_bound(guide_bbox.left, guide_width, source_width)
    top = _map_lower_bound(guide_bbox.top, guide_height, source_height)
    right = _map_upper_bound(guide_bbox.right, guide_width, source_width)
    bottom = _map_upper_bound(guide_bbox.bottom, guide_height, source_height)
    return CropRegion(left, top, right, bottom)


def _map_lower_bound(value: int, guide_limit: int, source_limit: int) -> int:
    """Map a guide coordinate down while keeping it within source bounds."""

    return min(source_limit - 1, max(0, floor(value * source_limit / guide_limit)))


def _map_upper_bound(value: int, guide_limit: int, source_limit: int) -> int:
    """Map a guide coordinate up while keeping a non-empty source extent."""

    return max(1, min(source_limit, ceil(value * source_limit / guide_limit)))


def _resize_local_mask(mask: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Scale only a retained mask's tight crop into source pixel coordinates."""

    return (
        functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(height, width),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .float()
    )
