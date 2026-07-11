# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Connected-component helpers for binary mask regions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain.segs import BoundingBox


@dataclass(frozen=True)
class MaskComponent:
    """Represent one connected mask component and its full-image bbox."""

    bbox: BoundingBox
    mask: torch.Tensor


def connected_mask_components(active_mask: torch.Tensor) -> tuple[MaskComponent, ...]:
    """Return 8-connected components from an HW active-pixel mask."""

    if active_mask.ndim != 2:
        raise ValueError("active_mask must be an HW tensor.")

    active = active_mask.detach().to(device="cpu", dtype=torch.bool)
    height = int(active.shape[0])
    width = int(active.shape[1])
    visited = torch.zeros((height, width), dtype=torch.bool)
    components: list[MaskComponent] = []

    for top in range(height):
        for left in range(width):
            if visited[top, left].item() or not active[top, left].item():
                continue
            components.append(_trace_component(active, visited, left, top))

    return tuple(components)


def _trace_component(
    active: torch.Tensor,
    visited: torch.Tensor,
    start_left: int,
    start_top: int,
) -> MaskComponent:
    """Trace one 8-connected component from its first active pixel."""

    height = int(active.shape[0])
    width = int(active.shape[1])
    queue: list[tuple[int, int]] = [(start_top, start_left)]
    visited[start_top, start_left] = True
    pixels: list[tuple[int, int]] = []
    index = 0

    while index < len(queue):
        top, left = queue[index]
        index += 1
        pixels.append((top, left))
        for neighbor_top in range(max(0, top - 1), min(height, top + 2)):
            for neighbor_left in range(max(0, left - 1), min(width, left + 2)):
                if visited[neighbor_top, neighbor_left].item():
                    continue
                visited[neighbor_top, neighbor_left] = True
                if active[neighbor_top, neighbor_left].item():
                    queue.append((neighbor_top, neighbor_left))

    top_values = [top for top, _left in pixels]
    left_values = [left for _top, left in pixels]
    bbox = BoundingBox(
        min(left_values),
        min(top_values),
        max(left_values) + 1,
        max(top_values) + 1,
    )
    component_mask = torch.zeros((height, width), dtype=torch.bool)
    for top, left in pixels:
        component_mask[top, left] = True
    return MaskComponent(bbox=bbox, mask=component_mask)
