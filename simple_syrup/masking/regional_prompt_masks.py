# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mask preparation for full-context regional prompting."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ..domain.regional_mask_bank import RegionalMaskBank
from .detailer_masks import gaussian_feather_mask


def prepare_regional_mask_batch(mask: object, feather: int) -> torch.Tensor:
    """Return a validated and optionally feathered BHW mask batch."""

    _, conditioning = _prepare_regional_masks(mask, feather)
    return conditioning


def build_regional_mask_bank(
    mask: object,
    *,
    feather: int,
    canvas_height: int,
    canvas_width: int,
) -> RegionalMaskBank:
    """Build separate planning and conditioning masks on one latent canvas."""

    authored, feathered = _prepare_regional_masks(mask, feather)
    planning_masks = resize_regional_mask_batch(
        authored,
        height=canvas_height,
        width=canvas_width,
    )
    conditioning_masks = resize_regional_mask_batch(
        feathered,
        height=canvas_height,
        width=canvas_width,
    )
    return RegionalMaskBank(
        planning_masks=planning_masks,
        conditioning_masks=conditioning_masks,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


def _prepare_regional_masks(
    mask: object,
    feather: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized authored masks and a separate feathered tensor."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("regional prompting requires a torch MASK tensor.")
    if feather < 0:
        raise ValueError("region_mask_feather must be greater than or equal to 0.")

    working = mask.float()
    if working.ndim == 2:
        working = working.unsqueeze(0)
    if working.ndim != 3:
        raise ValueError("regional prompting requires an HW or BHW MASK tensor.")
    if int(working.shape[0]) < 1:
        raise ValueError("regional prompting requires at least one authored mask.")
    if int(working.shape[1]) < 1 or int(working.shape[2]) < 1:
        raise ValueError("regional masks must have non-empty height and width.")

    normalized = working.clamp(0.0, 1.0)
    conditioning = (
        normalized.to(device=normalized.device, dtype=normalized.dtype, copy=True)
        if feather == 0
        else gaussian_feather_mask(normalized, feather)
    )
    return normalized, conditioning


def resize_regional_mask_batch(
    mask_batch: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize BHW masks while preserving authored area during downscaling."""

    if height < 1 or width < 1:
        raise ValueError("regional mask target height and width must be positive.")
    if tuple(mask_batch.shape[1:]) == (height, width):
        return mask_batch
    source_height, source_width = map(int, mask_batch.shape[1:])
    downscaled_height = min(height, source_height)
    downscaled_width = min(width, source_width)
    working = mask_batch.unsqueeze(1)
    if (downscaled_height, downscaled_width) != (source_height, source_width):
        working = functional.interpolate(
            working,
            size=(downscaled_height, downscaled_width),
            mode="area",
        )
    if (downscaled_height, downscaled_width) != (height, width):
        working = functional.interpolate(
            working,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return working.squeeze(1)


def regional_mask(mask_batch: torch.Tensor, index: int) -> torch.Tensor:
    """Return one positional region as a singleton BHW mask."""

    if index < 0 or index >= int(mask_batch.shape[0]):
        raise IndexError(f"regional mask index {index} is out of range.")
    return mask_batch[index : index + 1]
