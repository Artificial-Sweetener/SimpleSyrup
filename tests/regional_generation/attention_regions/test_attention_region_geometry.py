# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test attention-native temporal shaping and soft SEGS construction."""

from __future__ import annotations

import torch

from simple_syrup.domain.attention_region_capture import (
    AttentionCaptureProfile,
    AttentionEvidenceMode,
    AttentionRegionControls,
)
from simple_syrup.domain.attention_region_maps import CapturedAttentionMap
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.services.attention_region_rendering import (
    ATTENTION_REGION_RENDERING_SERVICE,
)


def test_anima_concept_mode_uses_validated_probability_aggregation() -> None:
    """Keep Anima on its spatially faithful cross-attention evidence policy."""

    maps = (
        _map(
            "cat",
            [0.1, 0.8, 0.2, 0.7],
            0.2,
            family=RegionalModelFamily.ANIMA,
            concept_values=[0.8, 0.1, 0.7, 0.2],
            baseline=0.1,
        ),
        _map(
            "cat",
            [0.1, 0.9, 0.1, 0.2],
            0.7,
            family=RegionalModelFamily.ANIMA,
            concept_values=[0.9, 0.1, 0.8, 0.1],
            baseline=0.1,
        ),
    )
    concept_controls = _controls(
        strength=0.2,
        consensus=0.25,
        evidence_mode=AttentionEvidenceMode.CONCEPT,
    )
    raw_controls = _controls(
        strength=0.2,
        consensus=0.25,
        evidence_mode=AttentionEvidenceMode.RAW,
    )

    _concept_segs, concept_mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=concept_controls,
        height=2,
        width=2,
    )
    _raw_segs, raw_mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=raw_controls,
        height=2,
        width=2,
    )

    assert not torch.equal(concept_mask, raw_mask)
    assert concept_mask[0, 0, 0] > concept_mask[0, 0, 1]
    assert raw_mask[0, 0, 1] > raw_mask[0, 0, 0]


def test_anima_concept_mode_recovers_connected_exact_token_geometry() -> None:
    """Keep a raw-attention appendage attached to the semantic concept core."""

    raw = [0.0] * 5 + [1.0, 1.0, 0.8, 0.7, 0.0] + [0.0] * 5
    concept = [0.0] * 5 + [1.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 5
    maps = tuple(
        _map(
            "cat",
            raw,
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=concept,
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.5,
            consensus=0.25,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=3,
        width=5,
    )

    assert mask[0, 1, :4].count_nonzero().item() == 4
    assert mask.count_nonzero().item() == 4


def test_anima_concept_mode_rejects_detached_exact_token_noise() -> None:
    """Exclude raw-attention components that do not touch the semantic core."""

    maps = tuple(
        _map(
            "cat",
            [1.0, 1.0, 0.0, 0.0, 0.9],
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=[1.0, 1.0, 0.0, 0.0, 0.0],
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.5,
            consensus=0.25,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=1,
        width=5,
    )

    assert mask[0, 0, :2].count_nonzero().item() == 2
    assert mask[0, 0, 4].item() == 0.0


def test_anima_concept_mode_preserves_complete_expansive_anchored_geometry() -> None:
    """Keep the complete attached object while dropping its weak global bridge."""

    maps = tuple(
        _map(
            "mage staff",
            [1.0, 1.0, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2],
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=1,
        width=10,
    )

    assert mask[0, 0, :5].count_nonzero().item() == 5
    assert mask[0, 0, 5:].count_nonzero().item() == 0


def test_anima_concept_mode_favors_recall_when_expansion_is_ambiguous() -> None:
    """Keep attached geometry when no tighter support extends beyond the core."""

    maps = tuple(
        _map(
            "close subject",
            [1.0, 1.0, 0.8, 0.8, 0.8],
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=[1.0, 1.0, 0.0, 0.0, 0.0],
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=1,
        width=5,
    )

    assert mask.count_nonzero().item() == 5


def test_anima_concept_mode_rejects_weak_broad_field_around_compact_peak() -> None:
    """Keep a compact semantic peak without absorbing its weak connected field."""

    raw = [0.0] * 100
    for row in range(3, 7):
        for column in range(10):
            raw[row * 10 + column] = 0.2
    raw[44] = 1.0
    raw[45] = 1.0
    concept = [0.0] * 100
    concept[44] = 1.0
    concept[45] = 1.0
    maps = tuple(
        _map(
            "compact feature",
            raw,
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=concept,
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=0.85,
        ),
        height=10,
        width=10,
    )

    assert mask.count_nonzero().item() == 2


def test_anima_concept_mode_rejects_global_reference_geometry_for_local_core() -> None:
    """Do not expand localized evidence through a near-global exact-token field."""

    raw = [0.3] * 100
    concept = [0.0] * 100
    for row in range(3, 7):
        for column in range(10):
            raw[row * 10 + column] = 0.4
    for column in range(3, 8):
        concept[4 * 10 + column] = 1.0
    maps = tuple(
        _map(
            "localized feature",
            raw,
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=concept,
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=0.85,
        ),
        height=10,
        width=10,
    )

    assert mask.count_nonzero().item() == 5


def test_anima_concept_mode_prefers_concentrated_geometry_for_moderate_core() -> None:
    """Use stricter geometry consensus when a moderate core anchors a broad field."""

    raw = [0.0] * 100
    for row in range(3, 7):
        for column in range(10):
            raw[row * 10 + column] = 0.2
    concept = [0.0] * 100
    for column in range(10):
        raw[4 * 10 + column] = 1.0
        concept[4 * 10 + column] = 1.0
    maps = tuple(
        _map(
            "moderate compact feature",
            raw,
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=concept,
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=0.85,
        ),
        height=10,
        width=10,
    )

    assert mask.count_nonzero().item() == 10


def test_anima_concept_mode_tightens_compact_geometry_below_frame_threshold() -> None:
    """Use the stable core when a compact weak field occupies under 15% of a frame."""

    raw = [0.0] * 400
    for row in range(6, 13):
        for column in range(6, 13):
            raw[row * 20 + column] = 0.2
    concept = [0.0] * 400
    for row in range(8, 11):
        for column in range(8, 11):
            raw[row * 20 + column] = 1.0
            concept[row * 20 + column] = 1.0
    maps = tuple(
        _map(
            "compact sub-frame feature",
            raw,
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=concept,
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=0.85,
        ),
        height=20,
        width=20,
    )

    assert mask.count_nonzero().item() == 9


def test_full_geometry_recall_preserves_weak_broad_field_around_compact_peak() -> None:
    """Let an explicit maximum-recall choice bypass adaptive compactness."""

    raw = [0.0] * 100
    for row in range(3, 7):
        for column in range(10):
            raw[row * 10 + column] = 0.2
    raw[44] = 1.0
    raw[45] = 1.0
    concept = [0.0] * 100
    concept[44] = 1.0
    concept[45] = 1.0
    maps = tuple(
        _map(
            "compact feature",
            raw,
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=concept,
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=1.0,
        ),
        height=10,
        width=10,
    )

    assert mask.count_nonzero().item() == 40


def test_anima_concept_mode_tightens_peak_dominated_semantic_field() -> None:
    """Tighten broad support when strength belongs mainly to a small core."""

    values = [0.05] * 100
    for row in range(3, 7):
        for column in range(10):
            values[row * 10 + column] = 0.3
    values[44] = 1.0
    values[45] = 1.0
    maps = tuple(
        _map(
            "compact semantic feature",
            values,
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=values,
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=0.85,
        ),
        height=10,
        width=10,
    )

    assert mask.count_nonzero().item() == 2


def test_anima_concept_mode_preserves_broad_high_strength_semantic_region() -> None:
    """Keep a genuinely broad concept whose support remains strong across its area."""

    values = [0.05] * 100
    for row in range(2, 8):
        for column in range(10):
            values[row * 10 + column] = 0.8
    values[44] = 1.0
    values[45] = 1.0
    maps = tuple(
        _map(
            "broad semantic region",
            values,
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=values,
        )
        for progress in (0.2, 0.7)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=0.85,
        ),
        height=10,
        width=10,
    )

    assert mask.count_nonzero().item() == 60


def _map(
    label: str,
    values: list[float],
    progress: float,
    *,
    baseline: float = 0.0,
    layer: str = "layer",
    family: RegionalModelFamily = RegionalModelFamily.STANDARD_UNET,
    concept_values: list[float] | None = None,
) -> CapturedAttentionMap:
    """Create one compact square attention observation."""

    return CapturedAttentionMap(
        label,
        torch.tensor(values, dtype=torch.float16),
        progress,
        layer,
        uniform_probability=baseline,
        model_family=family,
        concept_values=(
            torch.tensor(concept_values, dtype=torch.float16)
            if concept_values is not None
            else None
        ),
    )


def _controls(
    *,
    start: float = 0.0,
    end: float = 1.0,
    strength: float = 0.35,
    consensus: float = 0.25,
    minimum_size: int = 1,
    keep_only: int = 0,
    combine: bool = False,
    solidity: float = 0.0,
    feather: int = 8,
    split: float = 0.0,
    instance_recall: float = 0.65,
    geometry_recall: float = 0.85,
    evidence_mode: AttentionEvidenceMode = AttentionEvidenceMode.CONCEPT,
) -> AttentionRegionControls:
    """Return representative balanced rendering controls."""

    return AttentionRegionControls(
        start,
        end,
        strength,
        consensus,
        split,
        minimum_size,
        AttentionCaptureProfile.BALANCED,
        instance_recall=instance_recall,
        geometry_recall=geometry_recall,
        keep_only=keep_only,
        combine_segs=combine,
        matte_solidity=solidity,
        edge_feather=feather,
        evidence_mode=evidence_mode,
    )
