# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove processed regional-attention chunk selection contracts."""

from __future__ import annotations

from uuid import uuid4

import pytest
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
from simple_syrup.domain.regional_attention_selection import (
    REGIONAL_ATTENTION_SELECTION_SERVICE,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraPlan
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank


@pytest.mark.parametrize(
    ("selectors", "expected"),
    [
        ([], []),
        ([0], [RegionalAttentionBranch.POSITIVE]),
        ([1], [RegionalAttentionBranch.NEGATIVE]),
        ([0, 1], [RegionalAttentionBranch.POSITIVE, RegionalAttentionBranch.NEGATIVE]),
        ([1, 0], [RegionalAttentionBranch.NEGATIVE, RegionalAttentionBranch.POSITIVE]),
        (
            [0, 0, 1, 0, 1, 1],
            [
                RegionalAttentionBranch.POSITIVE,
                RegionalAttentionBranch.POSITIVE,
                RegionalAttentionBranch.NEGATIVE,
                RegionalAttentionBranch.POSITIVE,
                RegionalAttentionBranch.NEGATIVE,
                RegionalAttentionBranch.NEGATIVE,
            ],
        ),
    ],
)
def test_processed_chunk_selection_follows_exact_comfy_order(
    selectors: list[int],
    expected: list[RegionalAttentionBranch],
) -> None:
    """Support CFG-disabled, reversed, repeated, and interleaved chunk sequences."""

    plan = _processed_plan()
    chunks = REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
        plan,
        cond_or_uncond=selectors,
        conditioning_uuids=_uuids_for_selectors(plan, selectors),
        sigma=0.5,
    )
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(selectors)))
    assert [chunk.branch for chunk in chunks] == expected
    assert all(
        len(chunk.regional_entries) == plan.mask_bank.region_count for chunk in chunks
    )


@pytest.mark.parametrize(
    ("selectors", "error", "message"),
    [
        (object(), TypeError, "list or tuple"),
        ([True], TypeError, "integer 0 or 1"),
        (["0"], TypeError, "integer 0 or 1"),
        ([2], ValueError, "observed 2"),
        ([-1], ValueError, "observed -1"),
    ],
)
def test_processed_chunk_selection_rejects_invalid_selectors(
    selectors: object,
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed instead of guessing conditional branch order."""

    with pytest.raises(error, match=message):
        REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
            _processed_plan(),
            cond_or_uncond=selectors,
            conditioning_uuids=(
                [uuid4()] * len(selectors) if isinstance(selectors, list) else []
            ),
            sigma=0.5,
        )


def test_processed_chunk_selection_preserves_all_active_base_entries() -> None:
    """Map repeated branch chunks to simultaneous base entries in declared order."""

    plan = _processed_plan()
    positive_base = ProcessedRegionalAttentionContext(
        0,
        None,
        (
            _entry(0, torch.full((1, 2, 3), 1.0), 0.25),
            _entry(1, torch.full((1, 2, 3), 5.0), 0.75),
        ),
    )
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            positive_base,
            plan.positive.regional_contexts,
        ),
        plan.negative,
        plan.mask_bank,
        plan.lora_plan,
    )
    chunks = REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
        plan,
        cond_or_uncond=[0, 0, 1],
        conditioning_uuids=[
            positive_base.entries[0].uuid,
            positive_base.entries[1].uuid,
            plan.negative.base_context.entries[0].uuid,
        ],
        sigma=0.5,
    )
    assert [chunk.base_entry.entry_index for chunk in chunks] == [0, 1, 0]
    assert [chunk.base_entry.strength for chunk in chunks] == [0.25, 0.75, 1.0]
    with pytest.raises(ValueError, match="does not identify exactly one"):
        REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
            plan,
            cond_or_uncond=[0, 0, 0],
            conditioning_uuids=[
                positive_base.entries[0].uuid,
                positive_base.entries[1].uuid,
                uuid4(),
            ],
            sigma=0.5,
        )


def _processed(
    conditioning_index: int,
    region_index: int | None,
    *,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Build one small valid processed context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (_entry(0, torch.full((1, 2, 3), value), 1.0),),
    )


def _entry(
    entry_index: int,
    tensor: torch.Tensor,
    strength: float,
) -> ProcessedRegionalAttentionEntry:
    """Build one unscheduled processed entry with an explicit Comfy identity."""

    return ProcessedRegionalAttentionEntry(
        entry_index,
        uuid4(),
        UNBOUNDED_CONDITIONING_SCHEDULE,
        tensor,
        strength,
    )


def _processed_plan() -> ProcessedRegionalAttentionPlan:
    """Build distinguishable positive and negative processed banks."""

    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _processed(0, None, value=1.0),
            (_processed(1, 0, value=2.0),),
        ),
        ProcessedRegionalAttentionBranch(
            _processed(0, None, value=-1.0),
            (_processed(1, 0, value=-2.0),),
        ),
        _mask_bank(1),
        RegionalLoraPlan(()),
    )


def _uuids_for_selectors(
    plan: ProcessedRegionalAttentionPlan,
    selectors: list[int],
) -> list[object]:
    """Return matching static UUIDs for arbitrary characterized chunk order."""

    return [
        (
            plan.positive.base_context.entries[0].uuid
            if selector == 0
            else plan.negative.base_context.entries[0].uuid
        )
        for selector in selectors
    ]


def _mask_bank(region_count: int) -> RegionalMaskBank:
    """Build one canonical bank with distinct tensor storage."""

    planning = torch.zeros((region_count, 4, 5), dtype=torch.float32)
    conditioning = torch.zeros_like(planning)
    return RegionalMaskBank(planning, conditioning, 5, 4)
