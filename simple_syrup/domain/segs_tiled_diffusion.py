# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build irregular, SEGS-guided latent tiles for tiled diffusion sampling."""

from __future__ import annotations

import torch

from .segs import NativeSegs, Segment, coerce_segment_mask, coerce_segs
from .semantic_tiled_diffusion import build_semantic_tiled_diffusion_plan
from .tiled_diffusion import TiledDiffusionPlan


def build_segs_guided_tiled_diffusion_plan(
    *,
    segs: object,
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
    tile_batch_size: int,
) -> TiledDiffusionPlan:
    """Build bounded sampling windows whose irregular cores follow supplied SEGS.

    Every latent pixel receives exactly one ownership core.  Each core is sampled
    through a rectangular window, while its local blend mask retains the irregular
    boundary and shares a feathered overlap with neighboring cores.
    """

    native_segs = coerce_segs(segs)
    validate_segs_aspect_ratio(native_segs, latent_height, latent_width)
    ownership_masks = segs_ownership_masks(
        native_segs,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    return build_semantic_tiled_diffusion_plan(
        ownership_masks=ownership_masks,
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        merge_across_masks=True,
    )


def validate_segs_aspect_ratio(
    segs: NativeSegs,
    latent_height: int,
    latent_width: int,
) -> None:
    """Reject SEGS that cannot describe the sampled latent's image proportions."""

    source_height, source_width = segs[0]
    source_ratio = source_width / source_height
    latent_ratio = latent_width / latent_height
    if abs(source_ratio - latent_ratio) / source_ratio <= 0.02:
        return
    raise ValueError(
        "SEGS-guided tiled diffusion requires SEGS to match the latent image "
        f"aspect ratio; SEGS is {source_height}x{source_width}, latent is "
        f"{latent_height}x{latent_width}."
    )


def segs_ownership_masks(
    segs: NativeSegs,
    *,
    latent_height: int,
    latent_width: int,
) -> tuple[torch.Tensor, ...]:
    """Resolve overlapping SEGS into a deterministic latent ownership partition."""

    source_height, source_width = segs[0]
    segment_masks = tuple(
        segment_mask_to_latent(
            segment,
            source_height=source_height,
            source_width=source_width,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        for segment in segs[1]
    )
    ranked_indexes = sorted(
        range(len(segment_masks)),
        key=lambda index: (
            int(segment_masks[index].sum().item()),
            -float(segs[1][index].confidence),
            index,
        ),
    )
    occupied = torch.zeros((latent_height, latent_width), dtype=torch.bool)
    ownership_masks: list[torch.Tensor] = []
    for index in ranked_indexes:
        owned = torch.logical_and(segment_masks[index], torch.logical_not(occupied))
        if bool(owned.any()):
            ownership_masks.append(owned)
        occupied = torch.logical_or(occupied, segment_masks[index])
    background = torch.logical_not(occupied)
    if bool(background.any()):
        ownership_masks.append(background)
    if ownership_masks:
        return tuple(ownership_masks)
    return (torch.ones((latent_height, latent_width), dtype=torch.bool),)


def segment_mask_to_latent(
    segment: Segment,
    *,
    source_height: int,
    source_width: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Restore one crop-local SEG mask and map it to a latent-space mask."""

    return (
        segment_weight_to_latent(
            segment,
            source_height=source_height,
            source_width=source_width,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        >= 0.5
    )


def segment_weight_to_latent(
    segment: Segment,
    *,
    source_height: int,
    source_width: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Project one crop-local SEG mask into latent space without binarizing it."""

    crop = segment.crop_region
    if (
        crop.left < 0
        or crop.top < 0
        or crop.right > source_width
        or crop.bottom > source_height
        or crop.width < 1
        or crop.height < 1
    ):
        raise ValueError(
            "SEGS-guided tiled diffusion requires every SEG crop_region to fit "
            "inside the SEGS header dimensions."
        )
    local_mask = coerce_segment_mask(segment).detach().cpu()
    latent_top, latent_bottom = _latent_sample_range(
        crop.top,
        crop.bottom,
        source_height,
        latent_height,
    )
    latent_left, latent_right = _latent_sample_range(
        crop.left,
        crop.right,
        source_width,
        latent_width,
    )
    latent_mask = torch.zeros((latent_height, latent_width), dtype=torch.float32)
    if latent_bottom <= latent_top or latent_right <= latent_left:
        return latent_mask
    sampled_rows = (
        torch.div(
            torch.arange(latent_top, latent_bottom) * source_height,
            latent_height,
            rounding_mode="floor",
        )
        - crop.top
    )
    sampled_columns = (
        torch.div(
            torch.arange(latent_left, latent_right) * source_width,
            latent_width,
            rounding_mode="floor",
        )
        - crop.left
    )
    sampled_mask = (
        local_mask.clamp(0.0, 1.0)
        .index_select(
            0,
            sampled_rows,
        )
        .index_select(1, sampled_columns)
    )
    latent_mask[latent_top:latent_bottom, latent_left:latent_right] = sampled_mask
    return latent_mask


def _latent_sample_range(
    source_start: int,
    source_end: int,
    source_limit: int,
    latent_limit: int,
) -> tuple[int, int]:
    """Return latent coordinates whose nearest samples fall in a source interval."""

    start = (source_start * latent_limit + source_limit - 1) // source_limit
    end = (source_end * latent_limit + source_limit - 1) // source_limit
    return max(0, min(latent_limit, start)), max(0, min(latent_limit, end))
