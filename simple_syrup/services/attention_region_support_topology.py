# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select semantic attention support from its normalized spatial topology."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from ..masking.mask_components import connected_mask_components

MINIMUM_BROAD_SUPPORT_COVERAGE = 0.15
REFERENCE_CORE_STRENGTH = 0.25
MAXIMUM_COMPACT_CORE_SHARE = 0.5
MAXIMUM_RELATIVE_SEMANTIC_CORE_SHARE = 0.75
MAXIMUM_CONCENTRATED_GEOMETRY_SHARE = 0.75
MINIMUM_DOMINANT_COMPONENT_SHARE = 0.75
MINIMUM_DOMINANT_BOUNDING_BOX_FILL = 0.35
MAXIMUM_DOMINANT_COMPONENT_ELONGATION = 1.75
SUPPORT_STRENGTH_THRESHOLDS = tuple(index / 40.0 for index in range(8, 40))

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SupportTopology:
    """Describe the dominant connected shape of one support field."""

    dominant_share: float
    dominant_strength_share: float
    dominant_bounding_box_fill: float
    dominant_elongation: float


class AttentionRegionSupportTopologyPolicy:
    """Tighten peak-dominated fields without shrinking genuinely broad evidence."""

    def reference_strength(self, minimum_strength: float) -> float:
        """Return the observation strength used to test for a compact stable core."""

        return max(minimum_strength, REFERENCE_CORE_STRENGTH)

    def prefers_concentrated_evidence(
        self,
        primary: torch.Tensor,
        concentrated: torch.Tensor,
        minimum_strength: float,
        *,
        adaptive: bool,
    ) -> bool:
        """Return whether stricter observation consensus reveals a compact core."""

        if not adaptive:
            return False
        primary_alpha = _normalize(primary)
        primary_support = (primary_alpha > 0.0) & (primary_alpha >= minimum_strength)
        primary_coverage = _coverage(primary_support)
        concentrated_alpha = _normalize(concentrated)
        concentrated_support = (concentrated_alpha > 0.0) & (
            concentrated_alpha >= self.reference_strength(minimum_strength)
        )
        concentrated_coverage = _coverage(concentrated_support)
        if concentrated_coverage <= 0.0:
            return False
        if primary_coverage > MINIMUM_BROAD_SUPPORT_COVERAGE:
            return concentrated_coverage < primary_coverage * MAXIMUM_COMPACT_CORE_SHARE
        if (
            concentrated_coverage
            >= primary_coverage * MAXIMUM_RELATIVE_SEMANTIC_CORE_SHARE
        ):
            return False
        topology = _support_topology(primary_support, primary_alpha)
        preferred = (
            topology.dominant_share >= MINIMUM_DOMINANT_COMPONENT_SHARE
            and topology.dominant_bounding_box_fill
            >= MINIMUM_DOMINANT_BOUNDING_BOX_FILL
            and topology.dominant_elongation <= MAXIMUM_DOMINANT_COMPONENT_ELONGATION
        )
        logger.debug(
            "Evaluated compact semantic attention topology: "
            "preferred=%s primary=%.3f concentrated=%.3f dominance=%.3f "
            "strength_dominance=%.3f bounding_box_fill=%.3f elongation=%.3f",
            preferred,
            primary_coverage,
            concentrated_coverage,
            topology.dominant_share,
            topology.dominant_strength_share,
            topology.dominant_bounding_box_fill,
            topology.dominant_elongation,
        )
        return preferred

    def prefers_concentrated_geometry(
        self,
        primary: torch.Tensor,
        concentrated: torch.Tensor,
        *,
        geometry_recall: float,
    ) -> bool:
        """Tighten weak blob-like expansion while preserving sparse or thin extent."""

        if geometry_recall >= 1.0:
            return False
        primary_coverage = _coverage(primary)
        concentrated_coverage = _coverage(concentrated)
        if (
            concentrated_coverage <= 0.0
            or concentrated_coverage
            >= primary_coverage * MAXIMUM_CONCENTRATED_GEOMETRY_SHARE
        ):
            return False
        if primary_coverage > MINIMUM_BROAD_SUPPORT_COVERAGE:
            return True
        topology = _support_topology(primary)
        return (
            topology.dominant_share >= MINIMUM_DOMINANT_COMPONENT_SHARE
            and topology.dominant_bounding_box_fill
            >= MINIMUM_DOMINANT_BOUNDING_BOX_FILL
            and topology.dominant_elongation <= MAXIMUM_DOMINANT_COMPONENT_ELONGATION
        )

    def select(
        self,
        alpha: torch.Tensor,
        minimum_strength: float,
        *,
        adaptive: bool,
    ) -> torch.Tensor:
        """Return support selected from strength and evidence concentration."""

        support = (alpha > 0.0) & (alpha >= minimum_strength)
        if not adaptive:
            return support
        support_coverage = _coverage(support)
        if support_coverage <= MINIMUM_BROAD_SUPPORT_COVERAGE:
            return support
        reference_strength = self.reference_strength(minimum_strength)
        reference = (alpha > 0.0) & (alpha >= reference_strength)
        reference_coverage = _coverage(reference)
        if (
            reference_coverage <= 0.0
            or reference_coverage >= support_coverage * MAXIMUM_COMPACT_CORE_SHARE
        ):
            return support
        target_coverage = max(
            MINIMUM_BROAD_SUPPORT_COVERAGE,
            reference_coverage,
        )
        for threshold in SUPPORT_STRENGTH_THRESHOLDS:
            if threshold <= minimum_strength:
                continue
            candidate = (alpha > 0.0) & (alpha >= threshold)
            candidate_coverage = _coverage(candidate)
            if 0.0 < candidate_coverage <= target_coverage:
                return candidate
        return support


def _coverage(support: torch.Tensor) -> float:
    """Return the fraction of spatial positions selected by support."""

    return float(support.count_nonzero().item()) / float(support.numel())


def _normalize(values: torch.Tensor) -> torch.Tensor:
    """Normalize one evidence field without amplifying an empty tensor."""

    maximum = values.amax()
    return values if maximum <= 0.0 else values / maximum


def _support_topology(
    support: torch.Tensor,
    strength: torch.Tensor | None = None,
) -> _SupportTopology:
    """Measure whether one field is dominated by a compact connected blob."""

    components = connected_mask_components(support)
    total_area = int(support.count_nonzero().item())
    if not components or total_area <= 0:
        return _SupportTopology(0.0, 0.0, 0.0, 0.0)
    dominant = max(
        components,
        key=lambda component: int(component.mask.count_nonzero().item()),
    )
    dominant_area = int(dominant.mask.count_nonzero().item())
    bounding_box_area = dominant.bbox.width * dominant.bbox.height
    shorter_side = min(dominant.bbox.width, dominant.bbox.height)
    longer_side = max(dominant.bbox.width, dominant.bbox.height)
    total_strength = (
        float(strength[support].sum().item()) if strength is not None else total_area
    )
    dominant_strength = (
        float(strength[dominant.mask].sum().item())
        if strength is not None
        else dominant_area
    )
    return _SupportTopology(
        dominant_area / total_area,
        dominant_strength / total_strength if total_strength > 0.0 else 0.0,
        dominant_area / bounding_box_area,
        longer_side / shorter_side,
    )


ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY = AttentionRegionSupportTopologyPolicy()
