# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive narrow shared coherence bands from regional token ownership."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True, slots=True)
class RegionalSelfAttentionCoherenceProfile:
    """Describe regional queries, shared keys, and global output blending."""

    query_owners: torch.Tensor
    shared_keys: torch.Tensor
    global_blend: torch.Tensor


class RegionalSelfAttentionCoherencePolicy:
    """Share local region boundaries while preserving owned interiors."""

    def __init__(
        self,
        *,
        radius: int = 1,
        interior_global_blend: float = 0.125,
    ) -> None:
        """Set boundary expansion and the global composition channel strength."""

        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            raise ValueError("Coherence radius must be a non-negative integer.")
        if (
            isinstance(interior_global_blend, bool)
            or not isinstance(interior_global_blend, int | float)
            or not 0.0 <= float(interior_global_blend) < 1.0
        ):
            raise ValueError(
                "Interior global blend must be in the inclusive range [0, 1)."
            )
        self._radius = radius
        self._interior_global_blend = float(interior_global_blend)

    def resolve(
        self,
        owners: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> RegionalSelfAttentionCoherenceProfile:
        """Return continuous global blending around locally shared boundaries."""

        if (
            not isinstance(owners, torch.Tensor)
            or owners.ndim != 2
            or owners.dtype != torch.long
        ):
            raise TypeError("Coherence owners must use integer B/Q layout.")
        if height < 1 or width < 1 or int(owners.shape[1]) != height * width:
            raise ValueError("Coherence owners must match active H/W geometry.")
        grid = owners.reshape(int(owners.shape[0]), height, width)
        boundary = torch.zeros_like(grid, dtype=torch.bool)
        horizontal = grid[:, :, :-1].ne(grid[:, :, 1:])
        boundary[:, :, :-1] |= horizontal
        boundary[:, :, 1:] |= horizontal
        vertical = grid[:, :-1, :].ne(grid[:, 1:, :])
        boundary[:, :-1, :] |= vertical
        boundary[:, 1:, :] |= vertical
        blend = torch.full_like(
            grid,
            self._interior_global_blend,
            dtype=torch.float32,
        )
        blend = torch.where(boundary, blend.new_ones(()), blend)
        shared = boundary | torch.zeros_like(boundary)
        for distance in range(1, self._radius + 1):
            expanded = (
                functional.max_pool2d(
                    boundary.unsqueeze(1).to(torch.float32),
                    kernel_size=(2 * distance) + 1,
                    stride=1,
                    padding=distance,
                )
                .squeeze(1)
                .to(torch.bool)
            )
            ring = expanded & ~shared
            blend = torch.where(
                ring,
                blend.new_full(
                    (),
                    (self._radius + 1 - distance) / (self._radius + 1),
                ),
                blend,
            )
            shared |= expanded
        uncovered = grid.eq(-1)
        shared |= uncovered
        blend = torch.where(uncovered, blend.new_ones(()), blend)
        return RegionalSelfAttentionCoherenceProfile(
            owners,
            shared.reshape_as(owners),
            blend.reshape_as(owners),
        )


REGIONAL_SELF_ATTENTION_COHERENCE_POLICY = RegionalSelfAttentionCoherencePolicy()
