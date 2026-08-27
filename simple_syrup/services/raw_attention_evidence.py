# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Aggregate captured probabilities without concept-isolation inference."""

from __future__ import annotations

import torch

from ..domain.attention_region_capture import AttentionRegionControls
from ..domain.attention_region_evidence import AttentionConceptEvidence
from ..domain.attention_region_maps import CapturedAttentionMap
from .attention_region_observation_projection import (
    ATTENTION_OBSERVATION_PROJECTION_SERVICE,
)
from .attention_region_support import ATTENTION_CONCEPT_SUPPORT_SERVICE


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
        normalized = resized / resized.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        strength = resized.mean(dim=0)
        strength = strength / strength.amax().clamp_min(1e-12)
        consensus = (normalized >= controls.minimum_strength).float().mean(dim=0)
        confidence = strength * consensus
        persistent = torch.where(
            consensus >= controls.minimum_consensus,
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
        support = self._support(alpha, controls.minimum_strength)
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


class AnimaConceptAttentionEvidencePolicy(RawAttentionEvidencePolicy):
    """Aggregate Anima's contextualized semantic-head probabilities."""

    def _values(self, observation: CapturedAttentionMap) -> torch.Tensor:
        """Return phrase agreement above its observation-local spatial baseline."""

        values = (
            observation.concept_values
            if observation.concept_values is not None
            else observation.values
        )
        values = values.float()
        return (values - values.mean()).clamp_min(0.0)

    def _support(
        self,
        alpha: torch.Tensor,
        minimum_strength: float,
    ) -> torch.Tensor:
        """Retain weak contextualized extent attached to a strong semantic core."""

        return ATTENTION_CONCEPT_SUPPORT_SERVICE.select(alpha, minimum_strength)


RAW_ATTENTION_EVIDENCE_POLICY = RawAttentionEvidencePolicy()
ANIMA_CONCEPT_ATTENTION_EVIDENCE_POLICY = AnimaConceptAttentionEvidencePolicy()
