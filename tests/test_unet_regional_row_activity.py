# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize standard-UNet authored regional activity by model row."""

from __future__ import annotations

from uuid import uuid4

import torch

from simple_syrup.domain.conditioning_schedule import (
    UNBOUNDED_CONDITIONING_SCHEDULE,
)
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.unet_regional_row_activity import (
    STANDARD_UNET_REGIONAL_ROW_ACTIVITY_RESOLVER,
)


def test_row_activity_keeps_absent_unconditional_region_on_base_path() -> None:
    """Activate authored positive rows without duplicating absent negative rows."""

    plan = _plan(positive_region_count=1, negative_region_count=0)
    contexts = _contexts(
        branches=(RegionalAttentionBranch.NEGATIVE, RegionalAttentionBranch.POSITIVE),
        region_strengths=((1.0, 1.0, 1.0, 1.0),),
        latent_batch_size=2,
    )

    activity = STANDARD_UNET_REGIONAL_ROW_ACTIVITY_RESOLVER.resolve(plan, contexts)

    assert activity.tolist() == [[False, False, True, True]]


def test_row_activity_supports_authored_negative_and_inactive_rows() -> None:
    """Respect independent negative authorship and entry schedule activity."""

    plan = _plan(positive_region_count=1, negative_region_count=1)
    contexts = _contexts(
        branches=(RegionalAttentionBranch.POSITIVE, RegionalAttentionBranch.NEGATIVE),
        region_strengths=((0.0, 1.0),),
        latent_batch_size=1,
    )

    activity = STANDARD_UNET_REGIONAL_ROW_ACTIVITY_RESOLVER.resolve(plan, contexts)

    assert activity.tolist() == [[False, True]]


def _contexts(
    *,
    branches: tuple[RegionalAttentionBranch, ...],
    region_strengths: tuple[tuple[float, ...], ...],
    latent_batch_size: int,
) -> BatchedRegionalAttentionContexts:
    """Return distinguishable aligned rows with declared regional strengths."""

    batch = len(branches) * latent_batch_size
    base = torch.arange(float(batch)).reshape(batch, 1, 1)
    return BatchedRegionalAttentionContexts(
        latent_batch_size,
        tuple(
            RegionalAttentionChunkBatch(
                index,
                branch,
                index * latent_batch_size,
                (index + 1) * latent_batch_size,
            )
            for index, branch in enumerate(branches)
        ),
        base,
        tuple(
            BatchedRegionalAttentionRegion(
                region_index,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        base.clone(),
                        strengths,
                    ),
                ),
            )
            for region_index, strengths in enumerate(region_strengths)
        ),
    )


def _plan(
    *,
    positive_region_count: int,
    negative_region_count: int,
) -> ProcessedRegionalAttentionPlan:
    """Return one generic plan with independently authored CFG regions."""

    masks = torch.ones((1, 1, 1))
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _context(0, None, 1.0),
            tuple(
                _context(index + 1, index, float(index + 2))
                for index in range(positive_region_count)
            ),
        ),
        ProcessedRegionalAttentionBranch(
            _context(0, None, -1.0),
            tuple(
                _context(index + 1, index, float(-(index + 2)))
                for index in range(negative_region_count)
            ),
        ),
        RegionalMaskBank(masks, masks.clone(), 1, 1),
        EMPTY_REGIONAL_LORA_PLAN,
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one unbounded single-entry processed context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                UNBOUNDED_CONDITIONING_SCHEDULE,
                torch.full((1, 1, 1), value),
                1.0,
            ),
        ),
    )
