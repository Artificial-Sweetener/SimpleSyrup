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
from simple_syrup.services.attention_region_matte import ATTENTION_MATTE_SERVICE
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


def test_empty_maps_return_correctly_sized_no_op_outputs() -> None:
    """Return empty SEGS and a zero mask for unsupported graph/model paths."""

    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(),
        controls=_controls(),
        height=5,
        width=7,
    )

    assert segs == ((5, 7), ())
    assert mask.shape == (1, 5, 7)
    assert mask.count_nonzero().item() == 0


def test_disconnected_attention_islands_become_separate_instances() -> None:
    """Expose retained disconnected objects as independently controllable SEGS."""

    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("outdoors", [1.0, 0.0, 1.0], 0.5),),
        controls=_controls(strength=0.5, consensus=0.0),
        height=1,
        width=3,
    )

    assert len(segs[1]) == 2
    assert tuple(segment.label for segment in segs[1]) == ("outdoors", "outdoors")
    assert mask.count_nonzero().item() == 2


def test_split_sensitivity_preserves_the_complete_concept_union() -> None:
    """Change instance separation without deleting moderate concept support."""

    maps = (_map("cat", [1.0, 0.4, 0.4, 0.4, 1.0], 0.5),)
    _unsplit_segs, unsplit = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(strength=0.3, consensus=0.0, split=0.0),
        height=1,
        width=5,
    )
    split_segs, split = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(strength=0.3, consensus=0.0, split=1.0),
        height=1,
        width=5,
    )

    assert torch.equal(split, unsplit)
    assert split.count_nonzero().item() == 5
    assert len(split_segs[1]) == 2


def test_cohesive_support_retains_a_moderate_body_around_a_strong_core() -> None:
    """Keep the complete attended silhouette instead of its sparse core pixels."""

    values = [
        0.0,
        0.2,
        0.2,
        0.2,
        0.0,
        0.2,
        0.4,
        0.6,
        0.4,
        0.2,
        0.2,
        0.6,
        1.0,
        0.6,
        0.2,
        0.2,
        0.4,
        0.6,
        0.4,
        0.2,
        0.0,
        0.2,
        0.2,
        0.2,
        0.0,
    ]

    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("hair", values, 0.5),),
        controls=_controls(strength=0.15, consensus=0.0, split=0.75),
        height=5,
        width=5,
    )

    assert mask.count_nonzero().item() == 21
    assert mask[0, 2, 2].item() == 1.0
    assert mask[0, 0, 1].item() > 0.0


def test_minimum_region_size_removes_each_pockmark_independently() -> None:
    """Discard a tiny island without rejecting or merging the valid component."""

    values = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("hair", values, 0.5),),
        controls=_controls(strength=0.5, consensus=0.0, minimum_size=2),
        height=3,
        width=3,
    )

    assert len(segs[1]) == 1
    assert segs[1][0].bbox == (0, 0, 2, 1)
    assert mask.count_nonzero().item() == 2


def test_keep_only_and_combine_apply_per_concept_without_changing_union() -> None:
    """Retain top instances and optionally package their union as one SEG."""

    maps = (_map("cat", [1.0, 1.0, 0.0, 0.8, 0.8], 0.5),)
    separate, separate_mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.5,
            consensus=0.0,
            keep_only=2,
            combine=False,
        ),
        height=1,
        width=5,
    )
    combined, combined_mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.5,
            consensus=0.0,
            keep_only=2,
            combine=True,
        ),
        height=1,
        width=5,
    )

    assert len(separate[1]) == 2
    assert len(combined[1]) == 1
    assert torch.equal(separate_mask, combined_mask)


def test_matte_solidity_flattens_interior_and_edge_feather_softens_boundary() -> None:
    """Preserve raw alpha at zero and make a solid feathered matte at one."""

    maps = (_map("dress", [0.0, 0.6, 1.0, 0.7, 0.0], 0.5),)
    _raw_segs, raw = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(strength=0.2, consensus=0.0, solidity=0.0),
        height=1,
        width=5,
    )
    _solid_segs, solid = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.2,
            consensus=0.0,
            solidity=1.0,
            feather=1,
        ),
        height=1,
        width=5,
    )

    assert raw[0, 0, 1].item() != raw[0, 0, 2].item()
    assert solid[0, 0, 2].item() == 1.0
    assert 0.0 < solid[0, 0, 0].item() < 1.0


def test_full_matte_solidity_fills_only_enclosed_holes() -> None:
    """Fill an interior attention gap without filling exterior-connected space."""

    ring = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    _segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("hair", ring, 0.5),),
        controls=_controls(
            strength=0.5,
            consensus=0.0,
            solidity=1.0,
            feather=0,
        ),
        height=3,
        width=3,
    )

    assert torch.equal(mask, torch.ones_like(mask))


def test_full_matte_solidity_closes_a_narrow_exterior_connected_channel() -> None:
    """Remove thin attention squiggles without filling broad exterior space."""

    support = torch.zeros(7, 7, dtype=torch.bool)
    support[1:6, 1:6] = True
    support[1:4, 3] = False

    matte = ATTENTION_MATTE_SERVICE.shape(
        alpha=support.float(),
        support=support,
        solidity=1.0,
        edge_feather=0,
    )

    assert matte[3, 3].item() == 1.0
    assert matte[0].count_nonzero().item() == 0
    assert matte[:, 0].count_nonzero().item() == 0


def test_full_matte_solidity_removes_a_winding_channel_from_a_large_opening() -> None:
    """Smooth thin branches while preserving the large excluded exterior area."""

    support = torch.ones(100, 100, dtype=torch.bool)
    support[55:, 25:75] = False
    support[35:55, 49:52] = False
    support[42:45, 42:52] = False

    matte = ATTENTION_MATTE_SERVICE.shape(
        alpha=support.float(),
        support=support,
        solidity=1.0,
        edge_feather=0,
    )

    assert matte[38, 50].item() == 1.0
    assert matte[43, 44].item() == 1.0
    assert matte[80, 50].item() == 0.0


def test_full_matte_solidity_never_erases_accepted_thin_support() -> None:
    """Keep every accepted pixel when topology cleanup adds cohesive support."""

    support = torch.zeros(100, 100, dtype=torch.bool)
    support[10:90, 50] = True

    matte = ATTENTION_MATTE_SERVICE.shape(
        alpha=support.float(),
        support=support,
        solidity=1.0,
        edge_feather=0,
    )

    assert torch.all(matte[support] == 1.0)


def test_highest_confidence_keeps_stronger_component() -> None:
    """Rank components from pre-normalized evidence rather than normalized maxima."""

    segs, _mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("cat", [1.0, 0.0, 0.55], 0.5),),
        controls=AttentionRegionControls(
            0.0,
            1.0,
            0.4,
            0.0,
            0.0,
            1,
            AttentionCaptureProfile.BALANCED,
            keep_only=1,
            keep_by="highest confidence",
        ),
        height=1,
        width=3,
    )

    assert len(segs[1]) == 1
    assert segs[1][0].bbox == (0, 0, 1, 1)


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
        keep_only=keep_only,
        combine_segs=combine,
        matte_solidity=solidity,
        edge_feather=feather,
        evidence_mode=evidence_mode,
    )
