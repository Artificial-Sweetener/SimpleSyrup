# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Aggregate temporal attention observations into concept evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from ..domain.attention_region_capture import (
    AttentionEvidenceMode,
    AttentionRegionControls,
)
from ..domain.attention_region_evidence import AttentionConceptEvidence
from ..domain.attention_region_maps import CapturedAttentionMap
from ..domain.regional_model_capabilities import RegionalModelFamily
from .concept_attention_evidence import CONCEPT_ATTENTION_EVIDENCE_POLICY
from .raw_attention_evidence import (
    ANIMA_CONCEPT_ATTENTION_EVIDENCE_POLICY,
    RAW_ATTENTION_EVIDENCE_POLICY,
)


class AttentionEvidencePolicy(Protocol):
    """Define one explicit attention aggregation strategy."""

    def aggregate(
        self,
        label: str,
        observations: tuple[CapturedAttentionMap, ...],
        controls: AttentionRegionControls,
        height: int,
        width: int,
    ) -> AttentionConceptEvidence:
        """Return one rendered concept evidence value."""


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
            _policy(controls, tuple(observations)).aggregate(
                label,
                tuple(observations),
                controls,
                height,
                width,
            )
            for label, observations in grouped.items()
        )


def _policy(
    controls: AttentionRegionControls,
    observations: tuple[CapturedAttentionMap, ...],
) -> AttentionEvidencePolicy:
    """Select the explicit model-family evidence aggregation owner."""

    if controls.evidence_mode is AttentionEvidenceMode.RAW:
        return RAW_ATTENTION_EVIDENCE_POLICY
    if observations[0].model_family is RegionalModelFamily.ANIMA:
        return ANIMA_CONCEPT_ATTENTION_EVIDENCE_POLICY
    return CONCEPT_ATTENTION_EVIDENCE_POLICY


ATTENTION_EVIDENCE_AGGREGATOR = AttentionEvidenceAggregator()
