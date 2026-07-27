# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shared regional conditioning assembly."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.runtime.regional_conditioning_companion import (
    attach_global_companion,
)
from simple_syrup.services.regional_conditioning_service import (
    RegionalConditioningService,
)


def _conditioning(name: str) -> list[list[object]]:
    """Return recognizable standard Comfy conditioning."""

    return [[name, {"source": name}]]


def test_normal_conditioning_remains_global_only() -> None:
    """Normal conditioning is copied and broadcast globally."""

    positive = _conditioning("global positive")
    negative = _conditioning("global negative")

    assembled_positive, assembled_negative = RegionalConditioningService().assemble(
        positive=positive,
        negative=negative,
        masks=torch.ones((2, 4, 4)),
        regional_prompt_weight=0.5,
        region_mask_feather=0,
    )

    assert assembled_positive == positive
    assert assembled_negative == negative
    assert assembled_positive is not positive
    assert assembled_positive[0][1] is not positive[0][1]


def test_batches_pair_global_and_regions_independently() -> None:
    """Positive and negative regional counts may differ without fallback reuse."""

    masks = torch.stack(
        [torch.zeros((3, 3)), torch.ones((3, 3)), torch.full((3, 3), 0.5)]
    )
    positive = ConditioningBatch(
        (
            _conditioning("positive global"),
            _conditioning("positive region 0"),
            _conditioning("positive region 1"),
        )
    )
    negative = ConditioningBatch(
        (_conditioning("negative global"), _conditioning("negative region 0"))
    )

    assembled_positive, assembled_negative = RegionalConditioningService().assemble(
        positive=positive,
        negative=negative,
        masks=masks,
        regional_prompt_weight=0.75,
        region_mask_feather=0,
    )

    assert [item[0] for item in assembled_positive] == [
        "positive global",
        "positive region 0",
        "positive region 1",
    ]
    assert [item[0] for item in assembled_negative] == [
        "negative global",
        "negative region 0",
    ]
    assert assembled_positive[0][1]["default"] is True
    assert assembled_negative[0][1]["default"] is True
    assert "mask" not in assembled_positive[0][1]
    assert "mask" not in assembled_negative[0][1]
    assert torch.equal(assembled_positive[1][1]["mask"], masks[0:1])
    assert torch.equal(assembled_positive[2][1]["mask"], masks[1:2])
    assert torch.equal(assembled_negative[1][1]["mask"], masks[0:1])
    assert assembled_positive[1][1]["mask_strength"] == 0.75
    assert assembled_positive[2][1]["mask_strength"] == 0.75
    assert assembled_negative[1][1]["mask_strength"] == 0.75
    assert assembled_positive[1][1]["set_area_to_bounds"] is False


def test_excess_positive_or_negative_regions_fail() -> None:
    """Either conditioning side rejects regional entries without masks."""

    service = RegionalConditioningService()
    masks = torch.ones((1, 2, 2))
    excessive = ConditioningBatch(
        (_conditioning("global"), _conditioning("one"), _conditioning("two"))
    )

    with pytest.raises(ValueError, match="positive conditioning contains 2"):
        service.assemble(
            positive=excessive,
            negative=_conditioning("negative"),
            masks=masks,
            regional_prompt_weight=0.5,
            region_mask_feather=0,
        )
    with pytest.raises(ValueError, match="negative conditioning contains 2"):
        service.assemble(
            positive=_conditioning("positive"),
            negative=excessive,
            masks=masks,
            regional_prompt_weight=0.5,
            region_mask_feather=0,
        )


def test_feathering_preserves_inputs_and_softens_regional_copy() -> None:
    """Optional feathering changes attached masks without mutating the input."""

    masks = torch.zeros((1, 9, 9))
    masks[:, 3:6, 3:6] = 1.0
    original = masks.clone()

    positive, _ = RegionalConditioningService().assemble(
        positive=ConditioningBatch((_conditioning("global"), _conditioning("region"))),
        negative=_conditioning("negative"),
        masks=masks,
        regional_prompt_weight=0.5,
        region_mask_feather=2,
    )

    attached = positive[1][1]["mask"]
    assert isinstance(attached, torch.Tensor)
    assert torch.equal(masks, original)
    assert not torch.equal(attached, original)
    assert bool(torch.any((attached > 0.0) & (attached < 1.0)))


def test_mask_composition_preserves_segment_lora_hook_metadata() -> None:
    """Global and regional hook groups survive standard mask composition."""

    global_hooks = object()
    regional_hooks = object()
    positive = ConditioningBatch(
        (
            [["global", {"hooks": global_hooks, "other": "global metadata"}]],
            [["region", {"hooks": regional_hooks, "other": "region metadata"}]],
        )
    )

    assembled, _ = RegionalConditioningService().assemble(
        positive=positive,
        negative=_conditioning("negative"),
        masks=torch.ones((1, 3, 3)),
        regional_prompt_weight=0.5,
        region_mask_feather=0,
    )

    assert assembled[0][1]["hooks"] is global_hooks
    assert assembled[0][1]["other"] == "global metadata"
    assert assembled[1][1]["hooks"] is regional_hooks
    assert assembled[1][1]["other"] == "region metadata"


@pytest.mark.parametrize(
    ("regional_prompt_weight", "expected_sources", "expected_strengths"),
    [
        (0.0, ["global", "hooked global"], [1.0]),
        (0.5, ["global", "hooked global", "region"], [0.5, 0.5]),
        (1.0, ["global", "region"], [1.0]),
    ],
)
def test_hooked_global_companion_keeps_lora_full_while_prompt_weight_changes(
    regional_prompt_weight: float,
    expected_sources: list[str],
    expected_strengths: list[float],
) -> None:
    """Global and local prompt shares use one regional LoRA model state."""

    regional = attach_global_companion(
        [["region", {"hooks": "regional hooks"}]],
        [["hooked global", {"hooks": "regional hooks"}]],
    )

    assembled, _ = RegionalConditioningService().assemble(
        positive=ConditioningBatch((_conditioning("global"), regional)),
        negative=_conditioning("negative"),
        masks=torch.ones((1, 2, 2)),
        regional_prompt_weight=regional_prompt_weight,
        region_mask_feather=0,
    )

    assert [item[0] for item in assembled] == expected_sources
    assert [item[1]["mask_strength"] for item in assembled[1:]] == expected_strengths
    assert all(item[1]["hooks"] == "regional hooks" for item in assembled[1:])
    assert all(
        "simple_syrup.regional_global_companion" not in item[1] for item in assembled
    )


def test_zero_regional_prompt_weight_returns_only_unchanged_global_entries() -> None:
    """The zero endpoint disables regional conditioning completely."""

    positive, negative = RegionalConditioningService().assemble(
        positive=ConditioningBatch((_conditioning("global"), _conditioning("region"))),
        negative=ConditioningBatch(
            (_conditioning("global negative"), _conditioning("region negative"))
        ),
        masks=torch.ones((1, 2, 2)),
        regional_prompt_weight=0.0,
        region_mask_feather=0,
    )

    assert positive == _conditioning("global")
    assert negative == _conditioning("global negative")
    assert "mask" not in positive[0][1]
    assert "mask" not in negative[0][1]


def test_full_regional_prompt_weight_complements_global_inside_mask() -> None:
    """The one endpoint removes global influence only inside solid coverage."""

    mask = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])
    positive, _ = RegionalConditioningService().assemble(
        positive=ConditioningBatch((_conditioning("global"), _conditioning("region"))),
        negative=_conditioning("negative"),
        masks=mask,
        regional_prompt_weight=1.0,
        region_mask_feather=0,
    )

    assert positive[0][1]["default"] is True
    assert "mask" not in positive[0][1]
    assert torch.equal(positive[1][1]["mask"], mask)
    assert positive[1][1]["mask_strength"] == 1.0


@pytest.mark.parametrize("regional_prompt_weight", [0.25, 0.5, 1.0])
def test_matched_global_negative_retains_full_regional_influence(
    regional_prompt_weight: float,
) -> None:
    """Global and fallback negative shares sum to full strength in a region."""

    negative = ConditioningBatch(
        (_conditioning("global negative"), _conditioning("global negative"))
    )

    _, assembled_negative = RegionalConditioningService().assemble(
        positive=ConditioningBatch(
            (_conditioning("global positive"), _conditioning("regional positive"))
        ),
        negative=negative,
        masks=torch.ones((1, 2, 2)),
        regional_prompt_weight=regional_prompt_weight,
        region_mask_feather=0,
    )

    assert assembled_negative[0][1]["default"] is True
    assert assembled_negative[1][1]["mask_strength"] == regional_prompt_weight
    assert torch.equal(
        assembled_negative[1][1]["mask"],
        torch.ones((1, 2, 2)),
    )


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan")])
def test_invalid_regional_prompt_weight_fails_before_mask_processing(
    weight: float,
) -> None:
    """The service rejects invalid influence before touching mask inputs."""

    with pytest.raises(ValueError, match="regional_prompt_weight"):
        RegionalConditioningService().assemble(
            positive=_conditioning("positive"),
            negative=_conditioning("negative"),
            masks=object(),
            regional_prompt_weight=weight,
            region_mask_feather=0,
        )
