# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Group nearby disconnected support that belongs to one spatial concept extent."""

from __future__ import annotations

from collections import defaultdict
from importlib import import_module

import torch


class AttentionComponentCohesionService:
    """Group qualifying support fragments before concept-instance ranking."""

    def group(self, supports: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Return evidence-preserving unions of spatially cohesive fragments."""

        if not supports:
            return ()
        shape = supports[0].shape
        if len(shape) != 2 or any(support.shape != shape for support in supports):
            raise ValueError("Attention component supports must share one HW shape.")
        if len(supports) == 1:
            return supports
        cpu_supports = tuple(
            support.detach().to(device="cpu", dtype=torch.bool) for support in supports
        )
        bounds = tuple(_active_bounds(support) for support in cpu_supports)
        radius = _cohesion_radius(int(shape[0]), int(shape[1]))
        parents = list(range(len(cpu_supports)))
        for first_index, first in enumerate(cpu_supports):
            for second_index in range(first_index + 1, len(cpu_supports)):
                if not _bounds_overlap(bounds[first_index], bounds[second_index]):
                    continue
                if _within_gap(
                    first,
                    cpu_supports[second_index],
                    bounds[first_index],
                    bounds[second_index],
                    radius,
                ):
                    _union(parents, first_index, second_index)

        grouped_indices: defaultdict[int, list[int]] = defaultdict(list)
        for index in range(len(cpu_supports)):
            grouped_indices[_root(parents, index)].append(index)
        device = supports[0].device
        grouped = tuple(
            torch.stack(tuple(cpu_supports[index] for index in indices)).any(dim=0)
            for indices in grouped_indices.values()
        )
        ordered = sorted(grouped, key=_spatial_key)
        return tuple(support.to(device=device) for support in ordered)


def _cohesion_radius(height: int, width: int) -> int:
    """Scale the maximum fragment gap conservatively with output resolution."""

    return max(2, min(16, round(min(height, width) * 0.01)))


def _active_bounds(active: torch.Tensor) -> tuple[int, int, int, int]:
    """Return left, top, right, and bottom bounds for one non-empty support."""

    coordinates = active.nonzero(as_tuple=False)
    if int(coordinates.shape[0]) < 1:
        raise ValueError("Attention component support must contain active pixels.")
    top = int(coordinates[:, 0].min().item())
    bottom = int(coordinates[:, 0].max().item()) + 1
    left = int(coordinates[:, 1].min().item())
    right = int(coordinates[:, 1].max().item()) + 1
    return left, top, right, bottom


def _bounds_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    """Return whether two disconnected supports share spatial extent on both axes."""

    return max(first[0], second[0]) < min(first[2], second[2]) and max(
        first[1], second[1]
    ) < min(first[3], second[3])


def _within_gap(
    first: torch.Tensor,
    second: torch.Tensor,
    first_bounds: tuple[int, int, int, int],
    second_bounds: tuple[int, int, int, int],
    radius: int,
) -> bool:
    """Return whether a small evidence-free gap separates two support fragments."""

    first_area = (first_bounds[2] - first_bounds[0]) * (
        first_bounds[3] - first_bounds[1]
    )
    second_area = (second_bounds[2] - second_bounds[0]) * (
        second_bounds[3] - second_bounds[1]
    )
    if second_area < first_area:
        first, second = second, first
        first_bounds = second_bounds
    height, width = int(first.shape[0]), int(first.shape[1])
    left = max(0, first_bounds[0] - radius)
    top = max(0, first_bounds[1] - radius)
    right = min(width, first_bounds[2] + radius)
    bottom = min(height, first_bounds[3] + radius)
    diameter = radius * 2 + 1
    cv2 = import_module("cv2")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    source = first[top:bottom, left:right].to(dtype=torch.uint8).numpy()
    target = second[top:bottom, left:right].numpy()
    dilated = cv2.dilate(source, kernel)
    return bool((dilated.astype(bool) & target).any())


def _root(parents: list[int], index: int) -> int:
    """Return one union-find root while compressing its traversed path."""

    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union(parents: list[int], first: int, second: int) -> None:
    """Join two fragment sets deterministically by their lower root index."""

    first_root = _root(parents, first)
    second_root = _root(parents, second)
    if first_root == second_root:
        return
    lower, higher = sorted((first_root, second_root))
    parents[higher] = lower


def _spatial_key(support: torch.Tensor) -> tuple[int, int]:
    """Return deterministic top-to-bottom, left-to-right ordering."""

    left, top, _right, _bottom = _active_bounds(support)
    return top, left


ATTENTION_COMPONENT_COHESION_SERVICE = AttentionComponentCohesionService()
