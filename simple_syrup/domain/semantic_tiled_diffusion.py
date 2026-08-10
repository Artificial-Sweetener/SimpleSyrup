# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build tiled diffusion plans from non-overlapping latent ownership masks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .tiled_diffusion import (
    LatentTile,
    TiledDiffusionPlan,
    batch_latent_tiles,
    build_tiled_diffusion_plan,
)


@dataclass(frozen=True)
class _OwnershipCore:
    """Represent one latent ownership region before sampling-window placement."""

    mask: torch.Tensor
    bounds: tuple[int, int, int, int]
    area: int


def build_semantic_tiled_diffusion_plan(
    *,
    ownership_masks: Sequence[torch.Tensor],
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
    tile_batch_size: int,
    merge_across_masks: bool,
) -> TiledDiffusionPlan:
    """Build bounded windows whose write weights follow ownership masks."""

    base_plan = build_tiled_diffusion_plan(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
    )
    normalized_masks = _validate_ownership_masks(
        ownership_masks,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    max_core_width = max(1, base_plan.tile_width - base_plan.overlap)
    max_core_height = max(1, base_plan.tile_height - base_plan.overlap)
    split_groups = tuple(
        tuple(
            split_core
            for split_core in _split_core(
                _core_from_mask(mask),
                max_width=max_core_width,
                max_height=max_core_height,
            )
        )
        for mask in normalized_masks
    )
    if merge_across_masks:
        cores = _merge_small_cores(
            tuple(core for group in split_groups for core in group),
            max_width=max_core_width,
            max_height=max_core_height,
        )
    else:
        cores = tuple(
            core
            for group in split_groups
            for core in _merge_small_cores(
                group,
                max_width=max_core_width,
                max_height=max_core_height,
            )
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
                for core in cores
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


def _validate_ownership_masks(
    masks: Sequence[torch.Tensor],
    *,
    latent_height: int,
    latent_width: int,
) -> tuple[torch.Tensor, ...]:
    """Return non-empty boolean masks that cover the complete latent canvas."""

    normalized: list[torch.Tensor] = []
    coverage = torch.zeros((latent_height, latent_width), dtype=torch.bool)
    for index, mask in enumerate(masks):
        if mask.ndim != 2 or tuple(mask.shape) != (latent_height, latent_width):
            raise ValueError(
                "Semantic ownership mask "
                f"{index} must match latent shape {latent_height}x{latent_width}."
            )
        boolean_mask = mask.detach().cpu().bool()
        if not bool(boolean_mask.any()):
            continue
        if bool(torch.logical_and(coverage, boolean_mask).any()):
            raise ValueError("Semantic ownership masks must not overlap.")
        normalized.append(boolean_mask)
        coverage = torch.logical_or(coverage, boolean_mask)
    if not normalized:
        raise ValueError("Semantic tiled diffusion requires non-empty ownership.")
    if not bool(coverage.all()):
        raise ValueError("Semantic ownership masks must cover the latent canvas.")
    return tuple(normalized)


def _split_core(
    core: _OwnershipCore,
    *,
    max_width: int,
    max_height: int,
) -> tuple[_OwnershipCore, ...]:
    """Recursively divide a core into balanced pieces within its tile budget."""

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
        raise ValueError("Unable to split an oversized semantic tile core.")
    return first, second


def _merge_small_cores(
    cores: tuple[_OwnershipCore, ...],
    *,
    max_width: int,
    max_height: int,
) -> tuple[_OwnershipCore, ...]:
    """Greedily combine nearby cores when one bounded window can hold both."""

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
    """Return a candidate whose combined bounds fit one ownership budget."""

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
        raise ValueError("Semantic tiled diffusion generated an empty tile weight.")
    return LatentTile(x, y, tile_width, tile_height, weight_mask)


def _clamp_window_start(center: float, window_size: int, limit: int) -> int:
    """Center a fixed sampling window while keeping it inside latent bounds."""

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


def _core_from_mask(mask: torch.Tensor) -> _OwnershipCore:
    """Build one core with bounds and area computed exactly once."""

    bounds = _mask_bounds(mask)
    if bounds is None:
        raise ValueError("Semantic tiled diffusion cannot use an empty core.")
    return _OwnershipCore(mask=mask, bounds=bounds, area=int(mask.sum().item()))


def _mask_bounds(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Return left, top, right, bottom bounds for a non-empty boolean mask."""

    y_coords, x_coords = torch.where(mask)
    if y_coords.numel() == 0:
        return None
    return (
        int(x_coords.min().item()),
        int(y_coords.min().item()),
        int(x_coords.max().item()) + 1,
        int(y_coords.max().item()) + 1,
    )
