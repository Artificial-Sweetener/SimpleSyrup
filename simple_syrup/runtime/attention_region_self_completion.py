# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Complete cross-attention seeds through bounded spatial self-attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention_region_logits import scaled_attention_logits

MAXIMUM_SELF_COMPLETION_TOKENS = 16384
MAXIMUM_SELF_COMPLETION_ANCHORS = 6
SELF_COMPLETION_INFLUENCE = 0.25
RELATED_COMPLETION_INFLUENCE = 0.5
RELATED_COMPLETION_GATE = 0.35


@dataclass(frozen=True, slots=True)
class SpatialSelfAttention:
    """Hold positive-branch self-attention projections for one spatial layer."""

    query: torch.Tensor
    key: torch.Tensor

    def __post_init__(self) -> None:
        """Require aligned batch-head spatial projections."""

        if (
            self.query.ndim != 4
            or self.key.shape != self.query.shape
            or int(self.query.shape[-2]) < 1
            or int(self.query.shape[-1]) < 1
        ):
            raise ValueError("Spatial self-attention projections must align as BHQD.")


class AttentionRegionSelfCompletion:
    """Extend compact concept seeds with sparse model-native attention anchors."""

    def complete(
        self,
        seed: torch.Tensor,
        self_attention: SpatialSelfAttention | None,
        spatial_height: int,
        spatial_width: int,
        *,
        related_seed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Complete exact evidence and gate contextual recall by object grouping."""

        if self_attention is None:
            return seed
        query = self_attention.query
        key = self_attention.key
        if (
            seed.ndim != 2
            or int(seed.shape[0]) != int(query.shape[0])
            or int(seed.shape[1]) != int(query.shape[-2])
            or spatial_height * spatial_width != int(seed.shape[1])
        ):
            return seed
        anchor_indices = _grid_anchor_indices(seed, spatial_height, spatial_width)
        batch, heads, _tokens, channels = query.shape
        anchors = torch.gather(
            query,
            2,
            anchor_indices.reshape(batch, 1, -1, 1).expand(
                batch,
                heads,
                -1,
                channels,
            ),
        )
        logits = scaled_attention_logits(anchors, key)
        affinity = torch.softmax(logits, dim=-1).mean(dim=1)
        anchor_strength = torch.gather(seed.float(), 1, anchor_indices)
        anchor_strength = anchor_strength / anchor_strength.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
        completion = (affinity * anchor_strength.unsqueeze(-1)).sum(dim=1)
        completion = completion / completion.amax(dim=1, keepdim=True).clamp_min(1e-12)
        peak = seed.amax(dim=1, keepdim=True)
        relative_seed = seed / peak.clamp_min(1e-12)
        extension = (completion - relative_seed).clamp_min(0.0)
        completed = (
            seed + extension.to(dtype=seed.dtype) * peak * SELF_COMPLETION_INFLUENCE
        )
        if related_seed is None or related_seed.shape != seed.shape:
            return completed
        related = related_seed.float()
        relative_related = related / related.amax(dim=1, keepdim=True).clamp_min(1e-12)
        related_extension = (relative_related - relative_seed).clamp_min(0.0)
        gate = (
            (completion - RELATED_COMPLETION_GATE) / (1.0 - RELATED_COMPLETION_GATE)
        ).clamp(
            0.0,
            1.0,
        )
        return completed + (
            related_extension.to(dtype=seed.dtype)
            * gate.to(dtype=seed.dtype)
            * peak
            * RELATED_COMPLETION_INFLUENCE
        )


def _grid_anchor_indices(
    seed: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Choose one strong spatial seed from each cell of a compact 3x3 grid."""

    spatial = seed.reshape(int(seed.shape[0]), height, width)
    anchors: list[torch.Tensor] = []
    grid_size = min(3, height, width)
    for row_index in range(grid_size):
        top = height * row_index // grid_size
        bottom = height * (row_index + 1) // grid_size
        for column_index in range(grid_size):
            left = width * column_index // grid_size
            right = width * (column_index + 1) // grid_size
            region = spatial[:, top:bottom, left:right]
            local = region.flatten(start_dim=1).argmax(dim=1)
            region_width = right - left
            row = torch.div(local, region_width, rounding_mode="floor") + top
            column = local.remainder(region_width) + left
            anchors.append(row * width + column)
    candidates = torch.stack(anchors, dim=1)
    if int(candidates.shape[1]) <= MAXIMUM_SELF_COMPLETION_ANCHORS:
        return candidates
    strengths = torch.gather(seed.float(), 1, candidates)
    strongest = strengths.topk(MAXIMUM_SELF_COMPLETION_ANCHORS, dim=1).indices
    return torch.gather(candidates, 1, strongest)


ATTENTION_REGION_SELF_COMPLETION = AttentionRegionSelfCompletion()
