# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recover exact token-attention geometry anchored to an isolated concept core."""

from __future__ import annotations

import torch

from ..domain.attention_region_evidence import AttentionConceptEvidence
from ..masking.mask_components import connected_mask_components

MAXIMUM_GEOMETRY_EXPANSION = 2.0


class AttentionRegionGeometryRecoveryService:
    """Admit raw-attention geometry only when it touches semantic concept support."""

    def recover(
        self,
        *,
        semantic: AttentionConceptEvidence,
        geometry: AttentionConceptEvidence,
    ) -> AttentionConceptEvidence:
        """Return semantic evidence extended by anchored exact-token components."""

        if semantic.label != geometry.label:
            raise ValueError("Recovered attention evidence labels must match.")
        if (
            semantic.alpha.shape != geometry.alpha.shape
            or semantic.support.shape != geometry.support.shape
            or semantic.confidence.shape != geometry.confidence.shape
        ):
            raise ValueError("Recovered attention evidence geometry must match.")
        anchored = torch.zeros_like(semantic.support)
        semantic_support = semantic.support.to(device="cpu")
        for component in connected_mask_components(geometry.support):
            if (component.mask & semantic_support).any().item():
                anchored |= component.mask.to(device=anchored.device)
        semantic_area = int(semantic.support.count_nonzero().item())
        anchored_area = int(anchored.count_nonzero().item())
        if anchored_area > round(semantic_area * MAXIMUM_GEOMETRY_EXPANSION):
            anchored.zero_()
        support = semantic.support | anchored
        alpha = torch.maximum(
            semantic.alpha,
            geometry.alpha * anchored.to(dtype=geometry.alpha.dtype),
        )
        confidence = torch.maximum(
            semantic.confidence,
            geometry.confidence * anchored.to(dtype=geometry.confidence.dtype),
        )
        return AttentionConceptEvidence(
            semantic.label,
            alpha * support.to(dtype=alpha.dtype),
            support,
            confidence,
        )


ATTENTION_REGION_GEOMETRY_RECOVERY_SERVICE = AttentionRegionGeometryRecoveryService()
