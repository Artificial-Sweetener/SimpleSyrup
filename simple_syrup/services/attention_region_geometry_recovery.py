# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recover exact token-attention geometry anchored to an isolated concept core."""

from __future__ import annotations

import logging

import torch

from ..domain.attention_region_evidence import AttentionConceptEvidence
from ..masking.mask_components import connected_mask_components
from .attention_region_support_topology import (
    ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY,
)

MAXIMUM_GLOBAL_GEOMETRY_COVERAGE = 0.65
MAXIMUM_LOCALIZED_SEMANTIC_COVERAGE = 0.15
MINIMUM_AMBIGUOUS_GEOMETRY_CONTRAST = 0.05
MAXIMUM_COMPACT_CORE_EXPANSION_RATIO = 8.0
MINIMUM_DISPROPORTIONATE_GEOMETRY_COVERAGE = 0.15
MINIMUM_ADAPTIVE_GEOMETRY_COVERAGE = 0.02
GEOMETRY_RECOVERY_THRESHOLDS = tuple(index / 20.0 for index in range(4, 20))

logger = logging.getLogger(__name__)


class AttentionRegionGeometryRecoveryService:
    """Admit raw-attention geometry only when it touches semantic concept support."""

    def recover(
        self,
        *,
        semantic: AttentionConceptEvidence,
        geometry: AttentionConceptEvidence,
        concentrated_geometry: AttentionConceptEvidence,
        minimum_strength: float,
        geometry_recall: float,
    ) -> AttentionConceptEvidence:
        """Return semantic evidence extended by anchored exact-token components."""

        if not (semantic.label == geometry.label == concentrated_geometry.label):
            raise ValueError("Recovered attention evidence labels must match.")
        if (
            semantic.alpha.shape != geometry.alpha.shape
            or semantic.support.shape != geometry.support.shape
            or semantic.confidence.shape != geometry.confidence.shape
            or semantic.alpha.shape != concentrated_geometry.alpha.shape
            or semantic.support.shape != concentrated_geometry.support.shape
            or semantic.confidence.shape != concentrated_geometry.confidence.shape
        ):
            raise ValueError("Recovered attention evidence geometry must match.")
        semantic_support = semantic.support.to(device="cpu")
        recovery_threshold = minimum_strength + (
            (1.0 - geometry_recall) * (1.0 - minimum_strength) * 0.25
        )
        recoverable = geometry.support & (geometry.alpha >= recovery_threshold)
        anchored = _anchored_support(recoverable, semantic_support)
        concentrated = _anchored_support(
            concentrated_geometry.support,
            semantic_support,
        )
        semantic_coverage = _coverage(semantic_support)
        concentrated_coverage = _coverage(concentrated_geometry.support)
        logger.debug(
            "Evaluated attention geometry recovery: label=%s semantic=%.3f "
            "primary=%.3f concentrated=%.3f concentrated_total=%.3f "
            "threshold=%.3f recall=%.3f",
            semantic.label,
            semantic_coverage,
            _coverage(anchored),
            _coverage(concentrated),
            concentrated_coverage,
            recovery_threshold,
            geometry_recall,
        )
        if (
            geometry_recall < 1.0
            and semantic_coverage <= MAXIMUM_LOCALIZED_SEMANTIC_COVERAGE
            and concentrated_coverage > MAXIMUM_GLOBAL_GEOMETRY_COVERAGE
        ):
            logger.debug(
                "Rejected non-localized exact-token geometry for %s: "
                "semantic=%.3f concentrated=%.3f recall=%.3f",
                semantic.label,
                semantic_coverage,
                concentrated_coverage,
                geometry_recall,
            )
            return semantic
        selected_geometry = geometry
        anchored_coverage = _coverage(anchored)
        if geometry_recall < 1.0:
            if ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_geometry(
                anchored,
                concentrated,
                geometry_recall=geometry_recall,
            ):
                logger.debug(
                    "Selected concentrated attention geometry for %s: "
                    "%.3f -> %.3f coverage",
                    semantic.label,
                    anchored_coverage,
                    _coverage(concentrated),
                )
                anchored = concentrated
                selected_geometry = concentrated_geometry
        anchored = _tighten_disproportionate_geometry(
            selected_geometry.alpha,
            anchored,
            semantic_support,
            geometry_recall,
        )
        anchored = _tighten_global_geometry(
            selected_geometry.alpha,
            anchored,
            semantic_support,
        )
        support = semantic.support | anchored
        alpha = torch.maximum(
            semantic.alpha,
            selected_geometry.alpha * anchored.to(dtype=selected_geometry.alpha.dtype),
        )
        confidence = torch.maximum(
            semantic.confidence,
            selected_geometry.confidence
            * anchored.to(dtype=selected_geometry.confidence.dtype),
        )
        return AttentionConceptEvidence(
            semantic.label,
            alpha * support.to(dtype=alpha.dtype),
            support,
            confidence,
        )


def _tighten_disproportionate_geometry(
    alpha: torch.Tensor,
    anchored: torch.Tensor,
    semantic_support: torch.Tensor,
    geometry_recall: float,
) -> torch.Tensor:
    """Reject weak broad fields around a disproportionately compact semantic core."""

    if geometry_recall >= 1.0:
        return anchored
    semantic_coverage = _coverage(semantic_support)
    anchored_coverage = _coverage(anchored)
    if semantic_coverage <= 0.0:
        return anchored
    if (
        anchored_coverage <= MINIMUM_DISPROPORTIONATE_GEOMETRY_COVERAGE
        or anchored_coverage <= semantic_coverage * MAXIMUM_COMPACT_CORE_EXPANSION_RATIO
    ):
        return anchored
    target_coverage = max(
        MINIMUM_ADAPTIVE_GEOMETRY_COVERAGE,
        semantic_coverage * MAXIMUM_COMPACT_CORE_EXPANSION_RATIO,
    )
    for threshold in GEOMETRY_RECOVERY_THRESHOLDS:
        candidate = alpha > threshold
        if _coverage(candidate) > target_coverage:
            continue
        tightened = _anchored_support(candidate, semantic_support)
        if tightened.any().item() and _coverage(tightened) <= target_coverage:
            return tightened
    return torch.zeros_like(anchored)


def _tighten_global_geometry(
    alpha: torch.Tensor,
    anchored: torch.Tensor,
    semantic_support: torch.Tensor,
) -> torch.Tensor:
    """Break weak global bridges while favoring the largest recoverable extent."""

    if _coverage(anchored) <= MAXIMUM_GLOBAL_GEOMETRY_COVERAGE:
        return anchored
    for threshold in GEOMETRY_RECOVERY_THRESHOLDS:
        candidate = alpha > threshold
        if _coverage(candidate) > MAXIMUM_GLOBAL_GEOMETRY_COVERAGE:
            continue
        tightened = _anchored_support(candidate, semantic_support)
        if (tightened & ~semantic_support).any().item():
            return tightened
    values = alpha[anchored]
    if (
        values.numel() > 1
        and float(values.float().std(unbiased=False).item())
        >= MINIMUM_AMBIGUOUS_GEOMETRY_CONTRAST
    ):
        return anchored
    return torch.zeros_like(anchored)


def _anchored_support(
    support: torch.Tensor,
    semantic_support: torch.Tensor,
) -> torch.Tensor:
    """Return complete support components touching the semantic concept core."""

    anchored = torch.zeros_like(semantic_support)
    for component in connected_mask_components(support):
        if (component.mask & semantic_support).any().item():
            anchored |= component.mask.to(device=anchored.device)
    return anchored


def _coverage(support: torch.Tensor) -> float:
    """Return the fraction of the output frame occupied by support."""

    return float(support.count_nonzero().item()) / float(support.numel())


ATTENTION_REGION_GEOMETRY_RECOVERY_SERVICE = AttentionRegionGeometryRecoveryService()
