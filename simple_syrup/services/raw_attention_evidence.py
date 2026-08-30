# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Aggregate captured probabilities without concept-isolation inference."""

from __future__ import annotations

import torch

from ..domain.attention_region_capture import AttentionRegionControls
from ..domain.attention_region_evidence import AttentionConceptEvidence
from ..domain.attention_region_maps import CapturedAttentionMap
from .attention_region_geometry_recovery import (
    ATTENTION_REGION_GEOMETRY_RECOVERY_SERVICE,
)
from .attention_region_observation_projection import (
    ATTENTION_OBSERVATION_PROJECTION_SERVICE,
)
from .attention_region_support_topology import (
    ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY,
)
from .concept_attention_evidence import CONCEPT_ATTENTION_EVIDENCE_POLICY


class RawAttentionEvidencePolicy:
    """Render a transparent arithmetic aggregate of captured model probabilities."""

    def aggregate(
        self,
        label: str,
        observations: tuple[CapturedAttentionMap, ...],
        controls: AttentionRegionControls,
        height: int,
        width: int,
    ) -> AttentionConceptEvidence:
        """Return raw probability strength with user-selected support thresholds."""

        resized = ATTENTION_OBSERVATION_PROJECTION_SERVICE.stack(
            observations,
            self._values,
            height,
            width,
        )
        return self._aggregate_resized(
            label,
            resized,
            controls.minimum_strength,
            controls.minimum_consensus,
            height,
            width,
        )

    def aggregate_with_reference(
        self,
        label: str,
        observations: tuple[CapturedAttentionMap, ...],
        controls: AttentionRegionControls,
        height: int,
        width: int,
        reference_strength: float,
    ) -> tuple[AttentionConceptEvidence, AttentionConceptEvidence]:
        """Return permissive and concentrated evidence from one projected stack."""

        resized = ATTENTION_OBSERVATION_PROJECTION_SERVICE.stack(
            observations,
            self._values,
            height,
            width,
        )
        primary = self._aggregate_resized(
            label,
            resized,
            controls.minimum_strength,
            controls.minimum_consensus,
            height,
            width,
        )
        reference = self._aggregate_resized(
            label,
            resized,
            reference_strength,
            controls.minimum_consensus,
            height,
            width,
        )
        return primary, reference

    def _aggregate_resized(
        self,
        label: str,
        resized: torch.Tensor,
        minimum_strength: float,
        minimum_consensus: float,
        height: int,
        width: int,
    ) -> AttentionConceptEvidence:
        """Aggregate one already-projected observation stack."""

        normalized = resized / resized.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        strength = resized.mean(dim=0)
        strength = strength / strength.amax().clamp_min(1e-12)
        consensus = (normalized >= minimum_strength).float().mean(dim=0)
        confidence = strength * consensus
        persistent = torch.where(
            consensus >= minimum_consensus,
            confidence,
            torch.zeros_like(confidence),
        )
        restored = ATTENTION_OBSERVATION_PROJECTION_SERVICE.restore(
            persistent,
            height,
            width,
        )
        maximum = restored.amax()
        alpha = restored if maximum <= 0.0 else restored / maximum
        support = self._support(alpha, minimum_strength)
        restored_confidence = ATTENTION_OBSERVATION_PROJECTION_SERVICE.restore(
            confidence,
            height,
            width,
        )
        return AttentionConceptEvidence(
            label,
            alpha * support,
            support,
            restored_confidence,
        )

    def _values(self, observation: CapturedAttentionMap) -> torch.Tensor:
        """Return the unmodified captured probability vector."""

        return observation.values

    def _support(
        self,
        alpha: torch.Tensor,
        minimum_strength: float,
    ) -> torch.Tensor:
        """Apply the literal raw-attention strength threshold."""

        return (alpha > 0.0) & (alpha >= minimum_strength)


RAW_ATTENTION_EVIDENCE_POLICY = RawAttentionEvidencePolicy()


class AnimaConceptAttentionEvidencePolicy(RawAttentionEvidencePolicy):
    """Isolate Anima semantic evidence and recover anchored token geometry."""

    def aggregate(
        self,
        label: str,
        observations: tuple[CapturedAttentionMap, ...],
        controls: AttentionRegionControls,
        height: int,
        width: int,
    ) -> AttentionConceptEvidence:
        """Anchor exact prompt-token geometry to the semantic concept core."""

        semantic = CONCEPT_ATTENTION_EVIDENCE_POLICY.aggregate_anima(
            label,
            observations,
            controls,
            height,
            width,
        )
        reference_strength = (
            ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.reference_strength(
                controls.minimum_strength
            )
        )
        geometry, concentrated_geometry = (
            RAW_ATTENTION_EVIDENCE_POLICY.aggregate_with_reference(
                label,
                observations,
                controls,
                height,
                width,
                reference_strength,
            )
        )
        return ATTENTION_REGION_GEOMETRY_RECOVERY_SERVICE.recover(
            semantic=semantic,
            geometry=geometry,
            concentrated_geometry=concentrated_geometry,
            minimum_strength=controls.minimum_strength,
            geometry_recall=controls.geometry_recall,
        )


ANIMA_CONCEPT_ATTENTION_EVIDENCE_POLICY = AnimaConceptAttentionEvidencePolicy()
