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


def test_concept_isolation_rejects_weak_disconnected_context() -> None:
    """Drop a weak contextual island while retaining raw inspection evidence."""

    maps = (_map("bangs", [1.0, 0.9, 0.0, 0.3, 0.3], 0.5),)
    concept_segs, _concept_mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(strength=0.15, consensus=0.0),
        height=1,
        width=5,
    )
    raw_segs, _raw_mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=maps,
        controls=_controls(
            strength=0.15,
            consensus=0.0,
            evidence_mode=AttentionEvidenceMode.RAW,
        ),
        height=1,
        width=5,
    )

    assert len(concept_segs[1]) == 1
    assert concept_segs[1][0].bbox == (0, 0, 2, 1)
    assert len(raw_segs[1]) == 2


def test_concept_isolation_retains_multiple_confident_regions() -> None:
    """Keep plural concept instances when each has substantial evidence."""

    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("cuffs", [1.0, 0.9, 0.0, 0.7, 0.65], 0.5),),
        controls=_controls(strength=0.15, consensus=0.0),
        height=1,
        width=5,
    )

    assert len(segs[1]) == 2
    assert mask.count_nonzero().item() == 4


def test_concept_isolation_retains_sparse_instances_by_peak_evidence() -> None:
    """Keep small repeated instances without rewarding a larger region for area."""

    values = [0.6] * 16 + [0.0, 1.0, 0.0, 0.7]
    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("falling petals", values, 0.5),),
        controls=_controls(strength=0.15, consensus=0.0),
        height=1,
        width=len(values),
    )

    assert len(segs[1]) == 3
    assert mask[0, 0, 17].item() > 0.0
    assert mask[0, 0, 19].item() > 0.0


def test_instance_recall_can_restrict_sparse_results_to_the_strongest_peak() -> None:
    """Let users remove weaker disconnected instances without an area heuristic."""

    values = [0.6] * 16 + [0.0, 1.0, 0.0, 0.7]
    segs, _mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("falling petals", values, 0.5),),
        controls=_controls(
            strength=0.15,
            consensus=0.0,
            instance_recall=0.2,
        ),
        height=1,
        width=len(values),
    )

    assert len(segs[1]) == 1
    assert segs[1][0].bbox == (17, 0, 18, 1)


def test_concept_isolation_does_not_prefer_a_tiny_sharp_island() -> None:
    """Keep broad supported evidence when a disconnected pixel peaks higher."""

    values = [0.4] * 25 + [0.0, 1.0]
    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("hair", values, 0.5),),
        controls=_controls(strength=0.15, consensus=0.0),
        height=1,
        width=len(values),
    )

    assert any(segment.bbox == (0, 0, 25, 1) for segment in segs[1])
    assert mask[0, 0, :25].count_nonzero().item() == 25


def test_concept_isolation_rejects_pockmarks_around_a_compact_dominant_region() -> None:
    """Keep one compact body when much smaller disconnected peaks surround it."""

    values = torch.zeros(11, 11)
    values[3:8, 3:8] = 0.6
    values[0:2, 0:2] = 0.8
    values[0, 10] = 1.0
    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("torso", values.flatten().tolist(), 0.5),),
        controls=_controls(strength=0.15, consensus=0.0, feather=0),
        height=11,
        width=11,
    )

    assert len(segs[1]) == 1
    assert segs[1][0].bbox == (3, 3, 8, 8)
    assert mask.count_nonzero().item() == 25


def test_full_instance_recall_preserves_pockmarks_around_a_compact_region() -> None:
    """Honor an explicit request to retain every supported disconnected instance."""

    values = torch.zeros(11, 11)
    values[3:8, 3:8] = 0.6
    values[0:2, 0:2] = 0.8
    values[0, 10] = 1.0
    segs, _mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("torso", values.flatten().tolist(), 0.5),),
        controls=_controls(
            strength=0.15,
            consensus=0.0,
            feather=0,
            instance_recall=1.0,
        ),
        height=11,
        width=11,
    )

    assert len(segs[1]) == 3


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


def test_keep_largest_groups_a_nearby_detached_concept_fragment() -> None:
    """Treat qualifying nearby support as one instance before top-N ranking."""

    values = torch.zeros((9, 9), dtype=torch.float32)
    values[1, 1:8] = 1.0
    values[7, 1:8] = 1.0
    values[1:8, 1] = 1.0
    values[4:6, 4:6] = 0.8

    segs, mask = ATTENTION_REGION_RENDERING_SERVICE.render(
        maps=(_map("cat", values.flatten().tolist(), 0.5),),
        controls=_controls(
            strength=0.5,
            consensus=0.0,
            keep_only=1,
            feather=0,
        ),
        height=9,
        width=9,
    )

    assert len(segs[1]) == 1
    assert mask[0, 4:6, 4:6].count_nonzero().item() == 4


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


def test_full_matte_solidity_preserves_a_narrow_exterior_connected_channel() -> None:
    """Flatten alpha without inventing support inside an exterior channel."""

    support = torch.zeros(7, 7, dtype=torch.bool)
    support[1:6, 1:6] = True
    support[1:4, 3] = False

    matte = ATTENTION_MATTE_SERVICE.shape(
        alpha=support.float(),
        support=support,
        solidity=1.0,
        edge_feather=0,
    )

    assert matte[3, 3].item() == 0.0
    assert matte[0].count_nonzero().item() == 0
    assert matte[:, 0].count_nonzero().item() == 0


def test_full_matte_solidity_preserves_a_winding_exterior_channel() -> None:
    """Keep exterior-connected exclusions regardless of their shape or width."""

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

    assert matte[38, 50].item() == 0.0
    assert matte[43, 44].item() == 0.0
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
