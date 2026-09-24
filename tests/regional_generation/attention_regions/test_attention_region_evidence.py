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


def test_geometry_recall_controls_faint_connected_exact_token_extent() -> None:
    """Let users trade faint attached geometry for a tighter semantic core."""

    maps = tuple(
        _map(
            "cat tail",
            [
                1.0,
                1.0,
                0.6,
                0.6,
                0.4,
                0.4,
                0.2,
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=[1.0, 1.0] + [0.0] * 14,
        )
        for progress in (0.2, 0.7)
    )

    _recalled_segs, recalled = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=1.0,
        ),
        height=1,
        width=16,
    )
    _tight_segs, tight = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            geometry_recall=0.0,
        ),
        height=1,
        width=16,
    )

    assert recalled.count_nonzero().item() == 8
    assert tight.count_nonzero().item() == 6


def test_anima_concept_mode_rejects_below_baseline_late_residue() -> None:
    """Do not normalize negligible late-step Anima residue into full support."""

    maps = tuple(
        _map(
            "cuffs",
            [0.01, 0.01, 0.01, 0.01],
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=[0.0010, 0.0012, 0.0011, 0.0010],
            baseline=0.01,
        )
        for progress in (0.75, 0.9)
    )

    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=2,
        width=2,
    )

    assert segs[1] == ()
    assert mask.count_nonzero().item() == 0


def test_anima_concept_mode_removes_a_broad_contextual_field() -> None:
    """Keep local lift without treating a broadly elevated phrase as full-frame."""

    maps = tuple(
        _map(
            "swept bangs",
            [0.1, 0.1, 0.1, 0.1],
            progress,
            family=RegionalModelFamily.ANIMA,
            concept_values=[0.11, 0.11, 0.11, 0.14],
            baseline=0.1,
        )
        for progress in (0.2, 0.6)
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.25,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=2,
        width=2,
    )

    assert mask.count_nonzero().item() == 1
    assert mask[0, 1, 1].item() == 1.0


def test_anima_concept_mode_prefers_resolved_mid_pass_evidence() -> None:
    """Prevent an endpoint observation from tying resolved middle evidence."""

    maps = (
        _map(
            "boots",
            [0.1, 0.1, 0.1, 0.1],
            0.0,
            family=RegionalModelFamily.ANIMA,
            concept_values=[0.9, 0.1, 0.1, 0.1],
            baseline=0.1,
            layer="shared",
        ),
        _map(
            "boots",
            [0.1, 0.1, 0.1, 0.1],
            0.5,
            family=RegionalModelFamily.ANIMA,
            concept_values=[0.1, 0.1, 0.1, 0.9],
            baseline=0.1,
            layer="shared",
        ),
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.4,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=2,
        width=2,
    )

    assert mask.count_nonzero().item() == 1
    assert mask[0, 1, 1].item() == 1.0


def test_concept_isolation_prefers_repeated_support_over_transient_peak() -> None:
    """Suppress one strong flash when stable observations localize elsewhere."""

    maps = (
        _map(
            "hair",
            [0.25, 0.25, 0.25, 0.95],
            0.1,
            baseline=0.25,
            layer="early",
        ),
        _map(
            "hair",
            [0.25, 0.78, 0.25, 0.25],
            0.4,
            baseline=0.25,
            layer="middle-a",
        ),
        _map(
            "hair",
            [0.25, 0.82, 0.25, 0.25],
            0.6,
            baseline=0.25,
            layer="middle-b",
        ),
        _map(
            "hair",
            [0.25, 0.76, 0.25, 0.25],
            0.8,
            baseline=0.25,
            layer="late",
        ),
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.2,
            consensus=0.4,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=2,
        width=2,
    )

    assert mask.count_nonzero().item() == 1
    assert mask[0, 0, 1].item() == 1.0


def test_concept_isolation_preserves_repeated_fine_layer_structure() -> None:
    """Keep layer-local fine detail without admitting a one-step transient."""

    maps = tuple(
        _map(
            "hair",
            [0.25, 0.85, 0.25, 0.25],
            progress,
            baseline=0.25,
            layer="coarse",
        )
        for progress in (0.2, 0.4, 0.6, 0.8)
    ) + (
        _map(
            "hair",
            [0.25, 0.85, 0.52, 0.48],
            0.4,
            baseline=0.25,
            layer="fine",
        ),
        _map(
            "hair",
            [0.25, 0.85, 0.55, 0.25],
            0.7,
            baseline=0.25,
            layer="fine",
        ),
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.5,
            feather=0,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=2,
        width=2,
    )

    assert mask[0, 1, 0].item() > 0.0
    assert mask[0, 1, 1].item() == 0.0


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
