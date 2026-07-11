# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared helpers for SEGS detector node outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..domain.segs import (
    BoundingBox,
    ImpactSegs,
    NativeSegs,
    Segment,
    limit_segs,
    sort_segs,
    to_impact_compatible_segs,
)
from ..masking.segs_mask_ops import (
    crop_image,
    crop_mask,
    crop_region_for_bbox,
    validate_single_image,
)


@dataclass(frozen=True)
class CombinedSegsResult:
    """Return one combined SEGS payload and its full-image mask."""

    segs: NativeSegs
    mask: torch.Tensor


@dataclass(frozen=True)
class FinalizedSegsOutput:
    """Return Impact-compatible SEGS and its paired output mask."""

    segs: ImpactSegs
    mask: torch.Tensor


def build_combined_segs_result(
    image: object,
    segs: NativeSegs,
    crop_factor: float,
) -> CombinedSegsResult:
    """Build one combined SEGS payload and a standard ComfyUI mask."""

    image_tensor = validate_single_image(image, "SEGS detector")
    header, segments = segs
    height, width = header
    combined_mask = combined_mask_from_segs(segs)
    mask_output = combined_mask.unsqueeze(0).to(dtype=torch.float32, device="cpu")
    if not segments or not torch.any(combined_mask > 0):
        return CombinedSegsResult(segs=(header, ()), mask=mask_output)

    y_coords, x_coords = torch.where(combined_mask > 0)
    bbox_left = int(torch.min(x_coords).item())
    bbox_top = int(torch.min(y_coords).item())
    bbox_right = min(width, int(torch.max(x_coords).item()) + 1)
    bbox_bottom = min(height, int(torch.max(y_coords).item()) + 1)
    if bbox_right <= bbox_left or bbox_bottom <= bbox_top:
        return CombinedSegsResult(segs=(header, ()), mask=mask_output)

    bbox = BoundingBox(bbox_left, bbox_top, bbox_right, bbox_bottom)
    crop_region = crop_region_for_bbox(
        bbox,
        image_height=height,
        image_width=width,
        crop_factor=crop_factor,
    )
    cropped_mask = crop_mask(combined_mask, crop_region).detach().clone()
    combined_segment = Segment(
        cropped_image=crop_image(image_tensor, crop_region).detach().clone(),
        cropped_mask=cropped_mask,
        confidence=max(segment.confidence for segment in segments),
        crop_region=crop_region,
        bbox=bbox,
        label="combined",
    )
    return CombinedSegsResult(segs=(header, (combined_segment,)), mask=mask_output)


def finalize_detector_segs_output(
    image: object,
    segs: NativeSegs,
    keep_only: int,
    keep_by: str,
    crop_factor: float,
    sort_order: str,
    combine_segs: bool,
    combined_builder: Callable[
        [object, NativeSegs, float],
        CombinedSegsResult,
    ] = build_combined_segs_result,
) -> FinalizedSegsOutput:
    """Apply shared detector-style SEGS output policy."""

    limited_segs = limit_segs(segs, keep_only, keep_by)
    sorted_segs = sort_segs(limited_segs, sort_order)
    combined = combined_builder(image, sorted_segs, crop_factor)
    output_segs = combined.segs if combine_segs else sorted_segs
    return FinalizedSegsOutput(
        segs=to_impact_compatible_segs(output_segs),
        mask=combined.mask,
    )


def combined_mask_from_segs(segs: NativeSegs) -> torch.Tensor:
    """Combine cropped segment masks into one full-image HW mask."""

    header, segments = segs
    height, width = header
    mask = torch.zeros((height, width), dtype=torch.float32)
    for segment in segments:
        cropped_mask = coerce_cropped_mask(segment)
        region = segment.crop_region
        existing = mask[region.top : region.bottom, region.left : region.right]
        mask[region.top : region.bottom, region.left : region.right] = torch.maximum(
            existing,
            cropped_mask.float().cpu(),
        )
    return mask.clamp(0.0, 1.0)


def coerce_cropped_mask(segment: Segment) -> torch.Tensor:
    """Return a crop-local HW mask tensor for a segment."""

    if isinstance(segment.cropped_mask, torch.Tensor):
        cropped_mask = segment.cropped_mask.float()
    else:
        cropped_mask = torch.as_tensor(segment.cropped_mask, dtype=torch.float32)
    if cropped_mask.ndim == 3 and int(cropped_mask.shape[0]) == 1:
        cropped_mask = cropped_mask.squeeze(0)
    if cropped_mask.ndim != 2:
        raise ValueError("Segment cropped_mask must be HW shaped.")
    expected_shape = (segment.crop_region.height, segment.crop_region.width)
    actual_shape = (int(cropped_mask.shape[0]), int(cropped_mask.shape[1]))
    if actual_shape != expected_shape:
        raise ValueError("Segment cropped_mask must match its crop region.")
    return cropped_mask.clamp(0.0, 1.0)
