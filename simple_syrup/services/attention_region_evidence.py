# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Aggregate temporal attention observations into concept evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from ..domain.attention_geometry import factor_spatial_geometry
from ..domain.attention_region_capture import AttentionRegionControls
from ..domain.attention_region_maps import CapturedAttentionMap
from .attention_spatial_projection import ATTENTION_SPATIAL_PROJECTION_SERVICE


@dataclass(frozen=True, slots=True)
class AttentionConceptEvidence:
    """Hold one concept's alpha, support, and pre-matte confidence evidence."""

    label: str
    alpha: torch.Tensor
    support: torch.Tensor
    confidence: torch.Tensor


class AttentionEvidenceAggregator:
    """Combine selected observations without owning component or matte policy."""

    def aggregate(
        self,
        *,
        maps: tuple[CapturedAttentionMap, ...],
        controls: AttentionRegionControls,
        height: int,
        width: int,
        batch_index: int,
    ) -> tuple[AttentionConceptEvidence, ...]:
        """Return ordered concept evidence for one output batch member."""

        grouped: dict[str, list[CapturedAttentionMap]] = defaultdict(list)
        for attention_map in maps:
            if (
                attention_map.batch_index == batch_index
                and controls.capture_start
                <= attention_map.progress
                <= controls.capture_end
            ):
                grouped[attention_map.label].append(attention_map)
        return tuple(
            _aggregate_group(label, tuple(observations), controls, height, width)
            for label, observations in grouped.items()
        )


def _aggregate_group(
    label: str,
    observations: tuple[CapturedAttentionMap, ...],
    controls: AttentionRegionControls,
    height: int,
    width: int,
) -> AttentionConceptEvidence:
    """Combine strength and persistence for one readable concept."""

    resized = torch.stack(
        tuple(_resize_observation(value, height, width) for value in observations)
    )
    normalized = resized / resized.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    weights = torch.tensor(
        [value.confidence for value in observations], dtype=resized.dtype
    ).reshape(-1, 1, 1)
    strength = (resized * weights).sum(dim=0) / weights.sum().clamp_min(1e-6)
    strength = strength / strength.amax().clamp_min(1e-12)
    consensus = (normalized >= controls.minimum_strength).float().mean(dim=0)
    confidence = strength * consensus
    persistent = torch.where(
        consensus >= controls.minimum_consensus,
        confidence,
        torch.zeros_like(confidence),
    )
    maximum = persistent.amax()
    alpha = persistent if maximum <= 0.0 else persistent / maximum
    support = (alpha > 0.0) & (alpha >= controls.minimum_strength)
    return AttentionConceptEvidence(label, alpha * support, support, confidence)


def _resize_observation(
    observation: CapturedAttentionMap, height: int, width: int
) -> torch.Tensor:
    """Resize one map using captured geometry or a compatibility fallback."""

    source_height = observation.spatial_height
    source_width = observation.spatial_width
    if source_height is None or source_width is None:
        source_height, source_width = factor_spatial_geometry(
            int(observation.values.numel()), target_aspect=width / height
        )
    source = observation.values.float().reshape(1, 1, source_height, source_width)
    projected = ATTENTION_SPATIAL_PROJECTION_SERVICE.project(
        source[0, 0], observation.spatial_transforms
    )
    return functional.interpolate(
        projected.reshape(1, 1, int(projected.shape[0]), int(projected.shape[1])),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


ATTENTION_EVIDENCE_AGGREGATOR = AttentionEvidenceAggregator()
