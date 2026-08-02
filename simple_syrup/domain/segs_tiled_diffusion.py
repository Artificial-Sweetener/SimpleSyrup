# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build irregular, SEGS-guided latent tiles for tiled diffusion sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .segs import NativeSegs, Segment, coerce_segs
from .tiled_diffusion import (
    LatentTile,
    TiledDiffusionPlan,
    batch_latent_tiles,
    build_tiled_diffusion_plan,
)


@dataclass(frozen=True)
class _OwnershipCore:
    """Represent a non-overlapping latent ownership region before window placement."""

    mask: torch.Tensor
    bounds: tuple[int, int, int, int]
    area: int


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

    base_plan = build_tiled_diffusion_plan(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
    )
    native_segs = coerce_segs(segs)
    validate_segs_aspect_ratio(native_segs, latent_height, latent_width)
    ownership = _build_ownership_cores(
        native_segs,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    max_core_width = max(1, base_plan.tile_width - base_plan.overlap)
    max_core_height = max(1, base_plan.tile_height - base_plan.overlap)
    split_cores = tuple(
        split_core
        for core in ownership
        for split_core in _split_core(
            core,
            max_width=max_core_width,
            max_height=max_core_height,
        )
    )
    merged_cores = _merge_small_cores(
        split_cores,
        max_width=max_core_width,
        max_height=max_core_height,
    )
    tiles = tuple(
        sorted(
            (
                _tile_for_core(
                    core,
                    latent_width=latent_width,
                    latent_height=latent_height,
                    tile_width=base_plan.tile_width,
                    tile_height=base_plan.tile_height,
                    overlap=base_plan.overlap,
                )
                for core in merged_cores
            ),
            key=lambda tile: (tile.y, tile.x),
        )
    )
    batches, effective_batch_size = batch_latent_tiles(tiles, tile_batch_size)
    return TiledDiffusionPlan(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=base_plan.tile_width,
        tile_height=base_plan.tile_height,
        overlap=base_plan.overlap,
        requested_tile_batch_size=tile_batch_size,
        tile_batch_size=effective_batch_size,
        tiles=tiles,
        batches=batches,
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


def _build_ownership_cores(
    segs: NativeSegs,
    *,
    latent_height: int,
    latent_width: int,
) -> tuple[_OwnershipCore, ...]:
    """Resolve overlapping SEGS into one deterministic latent ownership partition."""

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
    cores: list[_OwnershipCore] = []
    for index in ranked_indexes:
        owned = torch.logical_and(segment_masks[index], torch.logical_not(occupied))
        if bool(owned.any()):
            cores.append(_core_from_mask(owned))
        occupied = torch.logical_or(occupied, segment_masks[index])
    background = torch.logical_not(occupied)
    if bool(background.any()):
        cores.append(_core_from_mask(background))
    if cores:
        return tuple(cores)
    return (
        _core_from_mask(torch.ones((latent_height, latent_width), dtype=torch.bool)),
    )


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
    local_mask = (
        torch.as_tensor(segment.cropped_mask, dtype=torch.float32).detach().cpu()
    )
    if local_mask.ndim == 3 and int(local_mask.shape[0]) == 1:
        local_mask = local_mask.squeeze(0)
    if local_mask.shape != (crop.height, crop.width):
        raise ValueError(
            "SEGS-guided tiled diffusion requires each cropped_mask to match its "
            "crop_region."
        )
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


def _split_core(
    core: _OwnershipCore,
    *,
    max_width: int,
    max_height: int,
) -> tuple[_OwnershipCore, ...]:
    """Recursively divide a core into balanced pieces that fit its tile budget."""

    left, top, right, bottom = core.bounds
    width = right - left
    height = bottom - top
    if width <= max_width and height <= max_height:
        return (core,)
    split_x = width / max_width >= height / max_height
    first, second = _split_mask_at_balanced_axis(core.mask, core.bounds, split_x)
    return _split_core(
        _core_from_mask(first), max_width=max_width, max_height=max_height
    ) + _split_core(_core_from_mask(second), max_width=max_width, max_height=max_height)


def _split_mask_at_balanced_axis(
    mask: torch.Tensor,
    bounds: tuple[int, int, int, int],
    split_x: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one non-empty mask near its active-pixel median on one axis."""

    left, top, right, bottom = bounds
    counts = (
        mask[top:bottom, left:right].sum(dim=0)
        if split_x
        else mask[top:bottom, left:right].sum(dim=1)
    )
    cumulative = torch.cumsum(counts, dim=0)
    midpoint = int(torch.searchsorted(cumulative, cumulative[-1] / 2, right=False))
    axis_start = left if split_x else top
    axis_end = right if split_x else bottom
    split_at = min(axis_end - 1, max(axis_start + 1, axis_start + midpoint + 1))
    first = mask.clone()
    second = mask.clone()
    if split_x:
        first[:, split_at:] = False
        second[:, :split_at] = False
    else:
        first[split_at:, :] = False
        second[:split_at, :] = False
    if not bool(first.any()) or not bool(second.any()):
        raise ValueError("Unable to split an oversized SEGS-guided tile core.")
    return first, second


def _merge_small_cores(
    cores: tuple[_OwnershipCore, ...],
    *,
    max_width: int,
    max_height: int,
) -> tuple[_OwnershipCore, ...]:
    """Greedily combine small nearby cores when one bounded window can hold both."""

    pending = list(cores)
    minimum_area = max(1, (max_width * max_height) // 4)
    merged = True
    while merged:
        merged = False
        for index, core in enumerate(tuple(pending)):
            if core.area >= minimum_area:
                continue
            candidate_index = _best_merge_candidate_index(
                core,
                pending,
                excluded_index=index,
                max_width=max_width,
                max_height=max_height,
            )
            if candidate_index is None:
                continue
            candidate = pending[candidate_index]
            pending[index] = _OwnershipCore(
                mask=torch.logical_or(core.mask, candidate.mask),
                bounds=_union_bounds(core.bounds, candidate.bounds),
                area=core.area + candidate.area,
            )
            pending.pop(candidate_index)
            merged = True
            break
    return tuple(pending)


def _best_merge_candidate_index(
    core: _OwnershipCore,
    candidates: list[_OwnershipCore],
    *,
    excluded_index: int,
    max_width: int,
    max_height: int,
) -> int | None:
    """Return a candidate index whose combined bounds fit one ownership budget."""

    eligible: list[tuple[int, int, int]] = []
    for index, candidate in enumerate(candidates):
        if index == excluded_index:
            continue
        bounds = _union_bounds(core.bounds, candidate.bounds)
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        if width > max_width or height > max_height:
            continue
        distance = _bounds_distance(core.bounds, candidate.bounds)
        eligible.append((width * height, distance, index))
    if not eligible:
        return None
    return min(eligible, key=lambda item: (item[0], item[1]))[2]


def _bounds_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    """Return the axis-aligned gap between two mask bounding boxes."""

    left, top, right, bottom = first
    other_left, other_top, other_right, other_bottom = second
    horizontal = max(0, other_left - right, left - other_right)
    vertical = max(0, other_top - bottom, top - other_bottom)
    return horizontal + vertical


def _union_bounds(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the tight rectangle containing both ownership-core bounds."""

    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _tile_for_core(
    core: _OwnershipCore,
    *,
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
) -> LatentTile:
    """Place one bounded sampling window around an irregular ownership core."""

    left, top, right, bottom = core.bounds
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    x = _clamp_window_start(center_x, tile_width, latent_width)
    y = _clamp_window_start(center_y, tile_height, latent_height)
    weight_mask = _feathered_tile_weight(
        core.mask,
        x=x,
        y=y,
        width=tile_width,
        height=tile_height,
        overlap=overlap,
    )
    if not bool((weight_mask > 0).any()):
        raise ValueError("SEGS-guided tiled diffusion generated an empty tile weight.")
    return LatentTile(x, y, tile_width, tile_height, weight_mask)


def _clamp_window_start(center: float, window_size: int, limit: int) -> int:
    """Center a fixed sampling window while keeping it inside the latent bounds."""

    desired = round(center - window_size / 2.0)
    return min(max(0, desired), limit - window_size)


def _feathered_tile_weight(
    mask: torch.Tensor,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    overlap: int,
) -> torch.Tensor:
    """Build one feathered tile weight without blurring the full latent mask."""

    if overlap == 0:
        return mask[y : y + height, x : x + width].float().contiguous()
    radius = max(1, overlap // 2)
    source_left = max(0, x - radius)
    source_top = max(0, y - radius)
    source_right = min(int(mask.shape[1]), x + width + radius)
    source_bottom = min(int(mask.shape[0]), y + height + radius)
    local_weight = (
        functional.avg_pool2d(
            mask[source_top:source_bottom, source_left:source_right]
            .float()
            .unsqueeze(0)
            .unsqueeze(0),
            kernel_size=radius * 2 + 1,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    local_y = y - source_top
    local_x = x - source_left
    return local_weight[
        local_y : local_y + height, local_x : local_x + width
    ].contiguous()


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


def _core_from_mask(mask: torch.Tensor) -> _OwnershipCore:
    """Build one core with bounds and area computed exactly once."""

    bounds = _mask_bounds(mask)
    if bounds is None:
        raise ValueError("SEGS-guided tiled diffusion cannot use an empty core.")
    return _OwnershipCore(
        mask=mask,
        bounds=bounds,
        area=int(mask.sum().item()),
    )


def _mask_bounds(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Return left, top, right, bottom bounds for one non-empty boolean mask."""

    y_coords, x_coords = torch.where(mask)
    if y_coords.numel() == 0:
        return None
    return (
        int(x_coords.min().item()),
        int(y_coords.min().item()),
        int(x_coords.max().item()) + 1,
        int(y_coords.max().item()) + 1,
    )
