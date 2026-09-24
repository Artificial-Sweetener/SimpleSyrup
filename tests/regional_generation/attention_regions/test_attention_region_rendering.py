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


def test_strength_and_consensus_shape_soft_regions_before_component_packaging() -> None:
    """Reject transient weak pixels while preserving feathered accepted values."""

    maps = (
        _map("hair", [0.1, 0.8, 0.2, 0.7], 0.2),
        _map("hair", [0.1, 0.9, 0.1, 0.2], 0.5),
        _map("hair", [0.1, 0.7, 0.1, 0.1], 0.8),
    )
    image = torch.zeros(1, 2, 2, 3)

    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(strength=0.5, consensus=0.66),
        height=2,
        width=2,
        image=image,
    )

    assert len(segs[1]) == 1
    assert segs[1][0].label == "hair"
    assert isinstance(segs[1][0].cropped_mask, torch.Tensor)
    assert segs[1][0].cropped_mask.shape == (1, 1)
    assert mask[0, 0, 1].item() == 1.0
    assert mask.count_nonzero().item() == 1


def test_temporal_window_changes_region_without_morphology() -> None:
    """Select early versus late attention evidence through normalized progress."""

    maps = (
        _map("composition", [1.0, 1.0, 0.0, 0.0], 0.1),
        _map("composition", [0.0, 0.0, 1.0, 1.0], 0.9),
    )

    _early_segs, early = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(start=0.0, end=0.4, strength=0.2, consensus=0.0),
        height=2,
        width=2,
    )
    _late_segs, late = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(start=0.6, end=1.0, strength=0.2, consensus=0.0),
        height=2,
        width=2,
    )

    assert early[0, 0].sum().item() > early[0, 1].sum().item()
    assert late[0, 1].sum().item() > late[0, 0].sum().item()


def test_raw_attention_preserves_diffuse_model_evidence() -> None:
    """Keep low-amplitude positive attention visible in inspection mode."""

    maps = (
        _map(
            "outfit",
            [0.26, 0.25, 0.27, 0.25],
            0.1,
            baseline=0.25,
        ),
        _map(
            "outfit",
            [0.25, 0.80, 0.25, 0.25],
            0.6,
            baseline=0.25,
        ),
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.0,
            consensus=0.0,
            evidence_mode=AttentionEvidenceMode.RAW,
        ),
        height=2,
        width=2,
    )

    assert mask.count_nonzero().item() == 4
    assert mask[0, 0, 1].item() == 1.0
    assert mask[0, 0, 0].item() > 0.0


def test_concept_isolation_removes_uniform_attention_baseline() -> None:
    """Do not promote near-uniform attention into concept support."""

    maps = (
        _map(
            "outfit",
            [0.26, 0.25, 0.27, 0.25],
            0.1,
            baseline=0.25,
        ),
        _map(
            "outfit",
            [0.25, 0.80, 0.25, 0.25],
            0.5,
            baseline=0.25,
        ),
        _map(
            "outfit",
            [0.25, 0.75, 0.25, 0.25],
            0.7,
            baseline=0.25,
        ),
    )

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.2,
            evidence_mode=AttentionEvidenceMode.CONCEPT,
        ),
        height=2,
        width=2,
    )

    assert mask.count_nonzero().item() == 1
    assert mask[0, 0, 1].item() == 1.0


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
