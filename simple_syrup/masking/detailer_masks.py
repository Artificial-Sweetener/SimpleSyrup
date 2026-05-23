# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detailer mask preparation helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

DETAILER_GAUSSIAN_SIGMA = 10.0


def gaussian_feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Soften a detailer mask edge with a Gaussian blur."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("detailer mask must be a torch.Tensor.")
    if radius < 0:
        raise ValueError("detailer mask radius must be greater than or equal to 0.")
    if mask.ndim not in {2, 3}:
        raise ValueError("detailer mask must be HW or BHW shaped.")

    working = mask.float().clamp(0.0, 1.0)
    if radius == 0 or _has_no_internal_boundary(working):
        return working

    was_hw = working.ndim == 2
    batched = working.unsqueeze(0) if was_hw else working
    kernel_size = _effective_kernel_size(
        radius,
        height=int(batched.shape[-2]),
        width=int(batched.shape[-1]),
    )
    if kernel_size == 0:
        return working
    effective_radius = kernel_size // 2
    samples = batched.unsqueeze(1)
    kernel = _gaussian_kernel(
        effective_radius,
        device=samples.device,
        dtype=samples.dtype,
    )
    padded_horizontal = F.pad(
        samples,
        (effective_radius, effective_radius, 0, 0),
        mode="replicate",
    )
    blurred = F.conv2d(padded_horizontal, kernel.view(1, 1, 1, -1))
    padded_vertical = F.pad(
        blurred,
        (0, 0, effective_radius, effective_radius),
        mode="replicate",
    )
    blurred = F.conv2d(padded_vertical, kernel.view(1, 1, -1, 1))
    result = blurred.squeeze(1).clamp(0.0, 1.0)
    return result.squeeze(0) if was_hw else result


def _has_no_internal_boundary(mask: torch.Tensor) -> bool:
    """Return whether every mask value is identical."""

    return bool(torch.all(mask == mask.flatten()[0]).item())


def _effective_kernel_size(radius: int, *, height: int, width: int) -> int:
    """Return a usable odd blur kernel size for a mask."""

    kernel_size = radius * 2 + 1
    shortest = min(height, width)
    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return 0
    return kernel_size


def _gaussian_kernel(
    radius: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalized one-dimensional Gaussian kernel."""

    positions = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=dtype,
    )
    kernel = torch.exp(-(positions * positions) / (2.0 * DETAILER_GAUSSIAN_SIGMA**2))
    return kernel / kernel.sum()
