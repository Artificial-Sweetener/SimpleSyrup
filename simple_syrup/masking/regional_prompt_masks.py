# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mask preparation for full-context regional prompting."""

from __future__ import annotations

import torch

from .detailer_masks import gaussian_feather_mask


def prepare_regional_mask_batch(mask: object, feather: int) -> torch.Tensor:
    """Return a validated and optionally feathered BHW mask batch."""

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
    if feather == 0:
        return normalized
    return gaussian_feather_mask(normalized, feather)


def regional_mask(mask_batch: torch.Tensor, index: int) -> torch.Tensor:
    """Return one positional region as a singleton BHW mask."""

    if index < 0 or index >= int(mask_batch.shape[0]):
        raise IndexError(f"regional mask index {index} is out of range.")
    return mask_batch[index : index + 1]
