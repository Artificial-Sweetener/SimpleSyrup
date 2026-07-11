# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Service for converting existing masks into image-associated native SEGS."""

from __future__ import annotations

import torch

from ..domain.segs import NativeSegs, Segment
from ..masking.mask_components import connected_mask_components
from ..masking.segs_mask_ops import (
    crop_image,
    crop_mask,
    crop_region_for_bbox,
    dilate_mask,
    validate_single_image,
)
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


class MaskToSEGSService:
    """Build separate native SEGS from one image and one existing mask."""

    def build(
        self,
        image: object,
        mask: object,
        mask_threshold: float,
        size_threshold: int,
        mask_dilation: int,
        post_dilation: int,
        crop_factor: float,
        label: str,
    ) -> NativeSegs:
        """Convert active mask components into image-associated SEGS."""

        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1.")
        if size_threshold < 1:
            raise ValueError("size_threshold must be at least 1.")

        image_tensor = validate_single_image(image, "Mask to SEGS")
        height = int(image_tensor.shape[1])
        width = int(image_tensor.shape[2])
        source_mask = _validate_single_mask(mask, height, width)
        working_mask = dilate_mask(source_mask, mask_dilation)
        active_mask = working_mask >= mask_threshold

        segments: list[Segment] = []
        for component in connected_mask_components(active_mask):
            if (
                component.bbox.width < size_threshold
                or component.bbox.height < size_threshold
            ):
                continue
            crop_region = crop_region_for_bbox(
                component.bbox,
                image_height=height,
                image_width=width,
                crop_factor=crop_factor,
            )
            component_values = torch.where(
                component.mask.to(device=working_mask.device),
                working_mask,
                torch.zeros_like(working_mask),
            )
            cropped_segment_mask = crop_mask(component_values, crop_region).detach()
            if post_dilation != 0:
                cropped_segment_mask = dilate_mask(cropped_segment_mask, post_dilation)
            segments.append(
                Segment(
                    cropped_image=crop_image(image_tensor, crop_region)
                    .detach()
                    .clone(),
                    cropped_mask=cropped_segment_mask.clone(),
                    confidence=1.0,
                    crop_region=crop_region,
                    bbox=component.bbox,
                    label=str(label),
                )
            )

        LOGGER.debug(
            "Built SEGS from mask",
            extra={
                "operation": "mask_to_segs",
                "mask_threshold": mask_threshold,
                "size_threshold": size_threshold,
                "mask_dilation": mask_dilation,
                "post_dilation": post_dilation,
                "crop_factor": crop_factor,
                "segment_count": len(segments),
            },
        )
        return (height, width), tuple(segments)


def _validate_single_mask(mask: object, height: int, width: int) -> torch.Tensor:
    """Return a validated crop-source HW mask tensor."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("Mask to SEGS requires a torch MASK tensor.")
    working = mask.float()
    if working.ndim == 3:
        batch_size = int(working.shape[0])
        if batch_size != 1:
            raise ValueError(
                "Mask to SEGS service requires one mask at a time; "
                f"received batch size {batch_size}."
            )
        working = working[0]
    if working.ndim != 2:
        raise ValueError("Mask to SEGS requires an HW or single-item BHW mask tensor.")
    actual_height = int(working.shape[0])
    actual_width = int(working.shape[1])
    if actual_height != height or actual_width != width:
        raise ValueError(
            "Mask to SEGS requires mask dimensions to match the image; "
            f"mask is {actual_height}x{actual_width}, image is {height}x{width}."
        )
    return working.clamp(0.0, 1.0)
