# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render deterministic SAM-style colored overlays from retained SEGS."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ..domain.segs import NativeSegs, Segment, coerce_segs

_REGION_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.26, 0.21),
    (0.13, 0.59, 0.95),
    (0.30, 0.69, 0.31),
    (1.00, 0.76, 0.03),
    (0.61, 0.15, 0.69),
    (1.00, 0.34, 0.13),
    (0.00, 0.74, 0.83),
    (0.91, 0.12, 0.39),
    (0.55, 0.76, 0.29),
    (0.40, 0.23, 0.72),
    (1.00, 0.60, 0.00),
    (0.00, 0.59, 0.53),
)


class SAMRegionOverlayRenderer:
    """Color retained SAM regions without changing their source SEGS geometry."""

    def render(self, *, image: torch.Tensor, segs: NativeSegs) -> torch.Tensor:
        """Return a source-sized IMAGE with translucent masks and stronger edges."""

        _validate_image(image)
        (segs_height, segs_width), segments = coerce_segs(segs)
        image_height = int(image.shape[1])
        image_width = int(image.shape[2])
        if (segs_height, segs_width) != (image_height, image_width):
            raise ValueError(
                "SAM region overlay requires SEGS dimensions to match the image."
            )
        if not segments:
            return image.detach().clone()

        color_channels = min(3, int(image.shape[-1]))
        device = image.device
        color_sum = torch.zeros(
            (image_height, image_width, color_channels),
            device=device,
            dtype=torch.float32,
        )
        coverage = torch.zeros(
            (image_height, image_width, 1),
            device=device,
            dtype=torch.float32,
        )
        boundaries = torch.zeros_like(coverage)
        boundary_thickness = max(1, round(max(image_height, image_width) / 1024))

        for index, segment in enumerate(segments):
            mask = _validated_local_mask(segment, device=device)
            region = segment.crop_region
            color = torch.tensor(
                _REGION_COLORS[index % len(_REGION_COLORS)][:color_channels],
                device=device,
                dtype=torch.float32,
            )
            mask_channels = mask.unsqueeze(-1)
            region_slice = (
                slice(region.top, region.bottom),
                slice(region.left, region.right),
            )
            color_sum[region_slice] += mask_channels * color
            coverage[region_slice] += mask_channels
            boundaries[region_slice] = torch.maximum(
                boundaries[region_slice],
                _mask_boundary(mask, thickness=boundary_thickness).unsqueeze(-1),
            )

        covered = coverage > 0
        mean_color = color_sum / coverage.clamp_min(1.0)
        alpha = torch.where(
            boundaries > 0,
            torch.full_like(coverage, 0.82),
            torch.full_like(coverage, 0.46),
        )
        alpha = torch.where(covered, alpha, torch.zeros_like(alpha))
        output = image.detach().clone()
        source_color = output[0, :, :, :color_channels].float()
        output[0, :, :, :color_channels] = (
            source_color * (1.0 - alpha) + mean_color * alpha
        ).to(dtype=output.dtype)
        return output.clamp(0.0, 1.0)


def _validate_image(image: torch.Tensor) -> None:
    """Reject tensors that cannot represent one ComfyUI image."""

    if image.ndim != 4 or int(image.shape[0]) != 1:
        raise ValueError("SAM region overlay requires one BHWC image.")
    if int(image.shape[-1]) < 1:
        raise ValueError("SAM region overlay requires at least one image channel.")


def _validated_local_mask(
    segment: Segment,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return one binary crop-local mask after validating its SEG geometry."""

    mask = segment.cropped_mask
    if not isinstance(mask, torch.Tensor):
        raise TypeError("SAM region overlay requires tensor SEG masks.")
    working = mask.detach().to(device=device, dtype=torch.float32)
    if working.ndim == 3 and int(working.shape[0]) == 1:
        working = working.squeeze(0)
    if working.ndim != 2:
        raise ValueError("SAM region overlay requires HW or 1HW SEG masks.")
    expected_shape = (segment.crop_region.height, segment.crop_region.width)
    if tuple(working.shape) != expected_shape:
        raise ValueError("SAM region overlay mask dimensions must match its SEG crop.")
    return (working >= 0.5).float()


def _mask_boundary(mask: torch.Tensor, *, thickness: int) -> torch.Tensor:
    """Return the interior boundary band of one binary mask."""

    kernel_size = thickness * 2 + 1
    padded = functional.pad(
        mask.unsqueeze(0).unsqueeze(0),
        (thickness, thickness, thickness, thickness),
        value=0.0,
    )
    eroded = -functional.max_pool2d(
        -padded,
        kernel_size=kernel_size,
        stride=1,
    )
    return (mask - eroded.squeeze(0).squeeze(0)).clamp(0.0, 1.0)
