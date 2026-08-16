# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own explicit-base Attention Couple parity weighting constants."""

from __future__ import annotations

import torch

GLOBAL_OWNERSHIP_STRENGTH = 0.6
REGIONAL_OWNERSHIP_STRENGTH = 0.4


def reference_base_mask(regional_masks: torch.Tensor) -> torch.Tensor:
    """Return PPM's explicit full-canvas global contribution."""

    if not isinstance(regional_masks, torch.Tensor):
        raise TypeError("Parity regional masks must be a torch.Tensor.")
    if regional_masks.ndim < 2 or int(regional_masks.shape[0]) < 1:
        raise ValueError("Parity regional masks require a leading region dimension.")
    if not regional_masks.is_floating_point():
        raise TypeError("Parity regional masks must use a floating dtype.")
    if not bool(torch.isfinite(regional_masks).all()):
        raise ValueError("Parity regional masks must contain finite values.")
    base_shape = tuple(int(size) for size in regional_masks.shape[1:])
    return regional_masks.new_full(
        base_shape,
        GLOBAL_OWNERSHIP_STRENGTH,
    )
