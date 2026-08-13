# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove base-only Attention Coupling sampler preparation and metadata admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import comfy.hooks
import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.raw_regional_attention import (
    build_raw_regional_attention_plan,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.services.attention_coupling_preparation_service import (
    ATTENTION_COUPLING_PREPARATION_SERVICE,
    AttentionCouplingMetadataError,
)


def test_preparation_returns_only_untouched_base_conditionings() -> None:
    """Keep regional contexts in the plan and out of ordinary KSampler inputs."""

    hooks = comfy.hooks.create_hook_lora({}, 1.0, 0.0)
    schedule = comfy.hooks.HookKeyframeGroup()
    schedule.add(comfy.hooks.HookKeyframe(0.0, 0.0, 0))
    schedule.add(comfy.hooks.HookKeyframe(1.0, 0.5, 0))
    hooks.set_keyframes_on_hooks(schedule)
    positive_metadata = {"pooled_output": torch.ones((1, 4)), "hooks": hooks}
    positive_base = _conditioning(
        1.0,
        positive_metadata,
    )
    positive_region = _conditioning(2.0, {"hooks": hooks})
    negative_base = _conditioning(-1.0, {"start_percent": 0.0, "end_percent": 1})
    negative_region = _conditioning(-2.0, {})
    plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch((positive_base, positive_region)),
        negative=ConditioningBatch((negative_base, negative_region)),
        mask_bank=_mask_bank(1),
    )

    prepared = ATTENTION_COUPLING_PREPARATION_SERVICE.prepare(plan)

    assert prepared.plan is plan
    assert prepared.positive is positive_base
    assert prepared.negative is negative_base
    assert prepared.plan.positive.regional_contexts[0].conditioning is positive_region
    assert prepared.plan.negative.regional_contexts[0].conditioning is negative_region
    assert positive_metadata["hooks"] is hooks
    pooled_output = positive_metadata["pooled_output"]
    assert isinstance(pooled_output, torch.Tensor)
    assert pooled_output.shape == (1, 4)


def test_preparation_reports_every_unsupported_metadata_field_together() -> None:
    """Reject all unsupported base and regional behavior before any patch stage."""

    positive_base = _conditioning(
        1.0,
        {
            "control": object(),
            "control_apply_to_uncond": True,
            "area": (1, 1, 0, 0),
        },
    )
    positive_region = _conditioning(
        2.0,
        {
            "mask": torch.ones((1, 2, 2)),
            "mask_strength": 0.5,
            "set_area_to_bounds": True,
        },
    )
    negative_base = _conditioning(-1.0, {"default": True, "gligen": object()})
    negative_region = _conditioning(-2.0, {"reference_latent": torch.ones((1,))})
    plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch((positive_base, positive_region)),
        negative=ConditioningBatch((negative_base, negative_region)),
        mask_bank=_mask_bank(1),
    )

    with pytest.raises(AttentionCouplingMetadataError) as captured:
        ATTENTION_COUPLING_PREPARATION_SERVICE.prepare(plan)

    assert [
        (issue.branch, issue.conditioning_index, issue.key)
        for issue in captured.value.issues
    ] == [
        ("positive", 0, "area"),
        ("positive", 0, "control"),
        ("positive", 0, "control_apply_to_uncond"),
        ("positive", 1, "mask"),
        ("positive", 1, "mask_strength"),
        ("positive", 1, "set_area_to_bounds"),
        ("negative", 0, "default"),
        ("negative", 0, "gligen"),
        ("negative", 1, "reference_latent"),
    ]
    message = str(captured.value)
    assert "positive conditioning 1" in message
    assert "negative conditioning 1" in message


@pytest.mark.parametrize(
    "metadata",
    [
        {"start_percent": 0.25},
        {"end_percent": 0.75},
        {"start_percent": 0.0, "end_percent": 1.0},
        {"start_percent": 0.0, "end_percent": 0.5},
        {"start_percent": 0.25, "end_percent": 0.75},
        {"start_percent": 0.5, "end_percent": 0.5},
    ],
)
def test_preparation_admits_valid_conditioning_schedules(
    metadata: dict[str, object],
) -> None:
    """Admit full, adjacent, overlapping, partial, and zero-width ranges."""

    plan = build_raw_regional_attention_plan(
        positive=_conditioning(1.0, metadata),
        negative=_conditioning(-1.0, {}),
        mask_bank=_mask_bank(1),
    )

    prepared = ATTENTION_COUPLING_PREPARATION_SERVICE.prepare(plan)

    assert prepared.plan is plan


@pytest.mark.parametrize(
    ("metadata", "key"),
    [
        ({"start_percent": float("nan")}, "start_percent"),
        ({"end_percent": True}, "end_percent"),
        ({"start_percent": -0.01}, "start_percent"),
        ({"end_percent": 1.01}, "end_percent"),
        ({"start_percent": 0.75, "end_percent": 0.25}, "<schedule>"),
    ],
)
def test_preparation_rejects_malformed_conditioning_schedules(
    metadata: dict[str, object],
    key: str,
) -> None:
    """Reject invalid authored ranges before installed Comfy conversion."""

    plan = build_raw_regional_attention_plan(
        positive=_conditioning(1.0, metadata),
        negative=_conditioning(-1.0, {}),
        mask_bank=_mask_bank(1),
    )

    with pytest.raises(AttentionCouplingMetadataError) as captured:
        ATTENTION_COUPLING_PREPARATION_SERVICE.prepare(plan)

    assert [issue.key for issue in captured.value.issues] == [key]


@pytest.mark.parametrize(
    ("conditioning", "key", "reason"),
    [
        ([], "<conditioning>", "non-empty"),
        ([object()], "<item>", "context tensor"),
        ([[object(), {}]], "<context>", "torch.Tensor"),
        ([[torch.ones((1, 1, 1)), object()]], "<metadata>", "dictionary"),
        ([[torch.ones((1, 1, 1)), {1: object()}]], "1", "keys must be strings"),
    ],
)
def test_preparation_rejects_malformed_standard_conditioning(
    conditioning: object,
    key: str,
    reason: str,
) -> None:
    """Reject malformed containers before Comfy processing or patch installation."""

    plan = build_raw_regional_attention_plan(
        positive=conditioning,
        negative=_conditioning(-1.0, {}),
        mask_bank=_mask_bank(1),
    )

    with pytest.raises(AttentionCouplingMetadataError) as captured:
        ATTENTION_COUPLING_PREPARATION_SERVICE.prepare(plan)

    assert any(
        issue.key == key and reason in issue.reason for issue in captured.value.issues
    )


def _conditioning(
    value: float,
    metadata: Mapping[Any, Any],
) -> list[list[object]]:
    """Build one standard Comfy conditioning entry."""

    return [[torch.full((1, 2, 3), value), metadata]]


def _mask_bank(region_count: int) -> RegionalMaskBank:
    """Build a canonical bank with distinct storage."""

    planning = torch.zeros((region_count, 4, 4), dtype=torch.float32)
    conditioning = torch.zeros_like(planning)
    return RegionalMaskBank(planning, conditioning, 4, 4)
