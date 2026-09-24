# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove sequence alignment is scoped to the current active model call."""

from __future__ import annotations

from uuid import uuid4

import torch

from simple_syrup.domain.conditioning_schedule import (
    UNBOUNDED_CONDITIONING_SCHEDULE,
    ConditioningScheduleRange,
)
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraPlan
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_attention_batching import (
    RegionalAttentionBatchingService,
)
from simple_syrup.runtime.regional_attention_sequence_alignment import (
    REGIONAL_ATTENTION_SEQUENCE_ALIGNER,
)


def test_inactive_region_does_not_expand_base_sequence() -> None:
    """Keep dormant authored contexts out of the native base-only call."""

    plan = _scheduled_plan()
    base_entry = plan.positive.base_context.entries[0]

    aligned = _service().align(
        plan,
        base_context=base_entry.cross_attention,
        cond_or_uncond=[0],
        conditioning_uuids=[base_entry.uuid],
        sigma=75.0,
        latent_batch_size=1,
    )

    assert aligned.base_context.shape == (1, 2, 4)
    assert aligned.base_context is base_entry.cross_attention
    assert aligned.regions[0].entries[0].context.shape == (1, 2, 4)
    assert aligned.regions[0].entries[0].strengths == (0.0,)


def test_active_region_expands_base_and_region_to_common_sequence() -> None:
    """Align the base only when the longer regional context participates."""

    plan = _scheduled_plan()
    base_entry = plan.positive.base_context.entries[0]

    aligned = _service().align(
        plan,
        base_context=base_entry.cross_attention,
        cond_or_uncond=[0],
        conditioning_uuids=[base_entry.uuid],
        sigma=25.0,
        latent_batch_size=1,
    )

    assert aligned.base_context.shape == (1, 6, 4)
    assert aligned.regions[0].entries[0].context.shape == (1, 6, 4)
    assert aligned.regions[0].entries[0].strengths == (1.0,)


def _service() -> RegionalAttentionBatchingService:
    """Return the standard-UNet batching configuration under test."""

    return RegionalAttentionBatchingService(
        sequence_aligner=REGIONAL_ATTENTION_SEQUENCE_ALIGNER
    )


def _scheduled_plan() -> ProcessedRegionalAttentionPlan:
    """Build one longer region that is inactive during the opening interval."""

    positive_base = _context(0, None, sequence_length=2)
    positive_region = _context(
        1,
        0,
        sequence_length=3,
        schedule=ConditioningScheduleRange(None, None, 50.0, 0.0),
    )
    negative_base = _context(0, None, sequence_length=2)
    mask = torch.ones((1, 2, 2))
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(positive_base, (positive_region,)),
        ProcessedRegionalAttentionBranch(negative_base, ()),
        RegionalMaskBank(mask, mask.clone(), 2, 2),
        RegionalLoraPlan(()),
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    *,
    sequence_length: int,
    schedule: ConditioningScheduleRange = UNBOUNDED_CONDITIONING_SCHEDULE,
) -> ProcessedRegionalAttentionContext:
    """Build one finite processed context with an explicit token length."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                schedule,
                torch.ones((1, sequence_length, 4)),
                1.0,
            ),
        ),
    )
