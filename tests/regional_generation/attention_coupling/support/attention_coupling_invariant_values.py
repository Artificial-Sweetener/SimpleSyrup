# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build canonical backend-neutral Attention Coupling invariant values."""

from __future__ import annotations

from uuid import uuid4

import torch

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank


def scheduled_invariant_plan() -> ProcessedRegionalAttentionPlan:
    """Return distinct CFG branches with one active and one inactive entry."""

    masks = torch.ones(1, 1, 1)
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _processed_context(0, None, ((1.0, None),)),
            (
                _processed_context(
                    1,
                    0,
                    (
                        (2.0, (0.75, 0.25)),
                        (8.0, (0.2, 0.1)),
                    ),
                ),
            ),
        ),
        ProcessedRegionalAttentionBranch(
            _processed_context(0, None, ((-1.0, None),)),
            (
                _processed_context(
                    1,
                    0,
                    (
                        (-2.0, (0.75, 0.25)),
                        (-8.0, (0.2, 0.1)),
                    ),
                ),
            ),
        ),
        RegionalMaskBank(masks, masks.clone(), 1, 1),
        EMPTY_REGIONAL_LORA_PLAN,
    )


def _processed_context(
    conditioning_index: int,
    region_index: int | None,
    values: tuple[tuple[float, tuple[float, float] | None], ...],
) -> ProcessedRegionalAttentionContext:
    """Build recognizable entries with optional converted-sigma bounds."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        tuple(
            ProcessedRegionalAttentionEntry(
                entry_index,
                uuid4(),
                (
                    ConditioningScheduleRange(None, None, None, None)
                    if bounds is None
                    else ConditioningScheduleRange(
                        None,
                        None,
                        bounds[0],
                        bounds[1],
                    )
                ),
                torch.full((1, 1, 1), value),
                1.0,
            )
            for entry_index, (value, bounds) in enumerate(values)
        ),
    )
