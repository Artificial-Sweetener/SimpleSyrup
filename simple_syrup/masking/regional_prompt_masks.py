# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mask preparation for full-context regional prompting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .detailer_masks import gaussian_feather_mask


@dataclass(frozen=True)
class PreparedRegionalMasks:
    """Keep authored geometry separate from feathered conditioning influence."""

    authored: torch.Tensor
    conditioning: torch.Tensor


def prepare_regional_mask_batch(mask: object, feather: int) -> torch.Tensor:
    """Return a validated and optionally feathered BHW mask batch."""

    return prepare_regional_masks(mask, feather).conditioning


def prepare_regional_masks(mask: object, feather: int) -> PreparedRegionalMasks:
    """Return normalized authored masks and their feathered conditioning form."""

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
        normalized if feather == 0 else gaussian_feather_mask(normalized, feather)
    )
    return PreparedRegionalMasks(
        authored=normalized,
        conditioning=conditioning,
    )


def resize_regional_mask_batch(
    mask_batch: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize BHW masks to one latent canvas using Comfy-compatible scaling."""

    if height < 1 or width < 1:
        raise ValueError("regional mask target height and width must be positive.")
    if tuple(mask_batch.shape[1:]) == (height, width):
        return mask_batch
    return functional.interpolate(
        mask_batch.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def regional_mask(mask_batch: torch.Tensor, index: int) -> torch.Tensor:
    """Return one positional region as a singleton BHW mask."""

    if index < 0 or index >= int(mask_batch.shape[0]):
        raise IndexError(f"regional mask index {index} is out of range.")
    return mask_batch[index : index + 1]
