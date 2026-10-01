# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project semantic regional-detailing ownership into each inversion resolution."""

from __future__ import annotations

import math

import torch.nn.functional as functional

from .regional_detailing import LatentBox, LatentRegion


def project_inversion_regions(
    regions: tuple[LatentRegion, ...],
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[LatentRegion, ...]:
    """Preserve region identity and conditioning while scaling masks and bounds."""
    if min(source_width, source_height, target_width, target_height) < 1:
        raise ValueError("Regional inversion canvases must have positive dimensions.")
    if (source_width, source_height) == (target_width, target_height):
        return regions
    projected: list[LatentRegion] = []
    for region in regions:
        box = region.latent_box
        if tuple(region.latent_mask.shape) != (source_height, source_width):
            raise ValueError("Regional inversion mask must match its canonical canvas.")
        if not (
            0 <= box.x < box.x + box.width <= source_width
            and 0 <= box.y < box.y + box.height <= source_height
        ):
            raise ValueError("Regional inversion bounds must remain inside the canvas.")
        left = math.floor(box.x * target_width / source_width)
        top = math.floor(box.y * target_height / source_height)
        right = math.ceil((box.x + box.width) * target_width / source_width)
        bottom = math.ceil((box.y + box.height) * target_height / source_height)
        mask = functional.interpolate(
            region.latent_mask[None, None].float(),
            size=(target_height, target_width),
            mode="nearest",
        )[0, 0].to(region.latent_mask)
        projected.append(
            LatentRegion(
                region.index,
                region.label,
                LatentBox(left, top, right - left, bottom - top),
                mask,
                region.positive,
            )
        )
    return tuple(projected)
