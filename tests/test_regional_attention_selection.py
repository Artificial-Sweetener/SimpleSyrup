# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact current-sigma regional conditioning entry selection."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
import torch

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention_selection import (
    REGIONAL_ATTENTION_SELECTION_SERVICE,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraPlan
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_attention_batching import (
    REGIONAL_ATTENTION_BATCHING_SERVICE,
)


@pytest.mark.parametrize(
    ("sigma", "active_indices"),
    [
        (100.0, (0,)),
        (75.0, (0, 2)),
        (50.0, (0, 1, 2, 3)),
        (25.0, (1, 2)),
        (0.0, (1,)),
    ],
)
def test_selection_preserves_all_active_entries_for_positive_and_negative_banks(
    sigma: float,
    active_indices: tuple[int, ...],
) -> None:
    """Keep inclusive adjacent, overlap, strength, identity, and source order."""

    plan = _scheduled_plan()
    positive_base = _active_base_entry(plan.positive, sigma=sigma)
    negative_base = _active_base_entry(plan.negative, sigma=sigma)

    chunks = REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
        plan,
        cond_or_uncond=[0, 1],
        conditioning_uuids=[positive_base.uuid, negative_base.uuid],
        sigma=sigma,
    )

    expected_positive = tuple(
        plan.positive.regional_contexts[0].entries[index] for index in active_indices
    )
    expected_negative = tuple(
        plan.negative.regional_contexts[0].entries[index] for index in active_indices
    )
    assert chunks[0].base_entry is positive_base
    assert chunks[1].base_entry is negative_base
    assert chunks[0].regional_entries == (expected_positive,)
    assert chunks[1].regional_entries == (expected_negative,)
    assert [entry.strength for entry in expected_positive] == [
        (0.1, 0.2, 0.3, 0.4)[index] for index in active_indices
    ]
    assert [entry.strength for entry in expected_negative] == [
        (0.1, 0.2, 0.3, 0.4)[index] for index in active_indices
    ]


def test_selection_uses_exact_uuid_when_adjacent_base_entries_are_both_active() -> None:
    """Resolve equality-boundary chunks by UUID instead of branch occurrence."""

    plan = _scheduled_plan()
    positive_entries = plan.positive.base_context.entries

    chunks = REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
        plan,
        cond_or_uncond=[0, 0],
        conditioning_uuids=[positive_entries[1].uuid, positive_entries[0].uuid],
        sigma=50.0,
    )

    assert [chunk.base_entry for chunk in chunks] == [
        positive_entries[1],
        positive_entries[0],
    ]


def test_selection_retains_authored_no_active_region_without_base_fallback() -> None:
    """Represent a schedule gap distinctly from an absent regional context."""

    plan = _gap_plan()
    uuids = [
        plan.positive.base_context.entries[0].uuid,
        plan.negative.base_context.entries[0].uuid,
    ]

    chunks = REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
        plan,
        cond_or_uncond=[0, 1],
        conditioning_uuids=uuids,
        sigma=50.0,
    )
    aligned = REGIONAL_ATTENTION_BATCHING_SERVICE.align(
        plan,
        base_context=torch.cat(
            (
                plan.positive.base_context.entries[0].cross_attention.repeat(2, 1, 1),
                plan.negative.base_context.entries[0].cross_attention.repeat(2, 1, 1),
            )
        ),
        cond_or_uncond=[0, 1],
        conditioning_uuids=uuids,
        sigma=50.0,
        latent_batch_size=2,
    )

    assert chunks[0].regional_entries == ((),)
    assert chunks[1].regional_entries == ((),)
    assert len(aligned.regions[0].entries) == 1
    assert aligned.regions[0].entries[0].strengths == (0.0, 0.0, 0.0, 0.0)


def test_selection_keeps_absent_region_as_explicit_base_fallback() -> None:
    """Distinguish no authored regional context from an inactive authored one."""

    plan = _gap_plan(negative_region=False)
    uuids = [
        plan.positive.base_context.entries[0].uuid,
        plan.negative.base_context.entries[0].uuid,
    ]

    chunks = REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
        plan,
        cond_or_uncond=[0, 1],
        conditioning_uuids=uuids,
        sigma=50.0,
    )
    aligned = REGIONAL_ATTENTION_BATCHING_SERVICE.align(
        plan,
        base_context=torch.cat(
            (
                plan.positive.base_context.entries[0].cross_attention,
                plan.negative.base_context.entries[0].cross_attention,
            )
        ),
        cond_or_uncond=[0, 1],
        conditioning_uuids=uuids,
        sigma=50.0,
        latent_batch_size=1,
    )

    assert chunks[0].regional_entries == ((),)
    assert chunks[1].regional_entries == (None,)
    assert aligned.regions[0].entries[0].strengths == (0.0, 1.0)


def test_selection_rejects_unknown_or_inactive_base_uuid() -> None:
    """Fail closed when supplied Comfy identity cannot be active in its branch."""

    plan = _scheduled_plan()
    inactive = plan.positive.base_context.entries[1]
    with pytest.raises(ValueError, match="inactive at sigma"):
        REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
            plan,
            cond_or_uncond=[0],
            conditioning_uuids=[inactive.uuid],
            sigma=75.0,
        )
    with pytest.raises(ValueError, match="does not identify exactly one"):
        REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
            plan,
            cond_or_uncond=[1],
            conditioning_uuids=[uuid4()],
            sigma=75.0,
        )
    equal_but_distinct_uuid = UUID(str(plan.negative.base_context.entries[0].uuid))
    with pytest.raises(ValueError, match="does not identify exactly one"):
        REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
            plan,
            cond_or_uncond=[1],
            conditioning_uuids=[equal_but_distinct_uuid],
            sigma=75.0,
        )


@pytest.mark.parametrize(
    ("uuids", "sigma", "error", "message"),
    [
        ([], 50.0, ValueError, "equal lengths"),
        ([object()], 50.0, TypeError, "Comfy UUID"),
        (None, 50.0, TypeError, "list or tuple"),
        ("valid", True, TypeError, "sigma must be a real"),
        ("valid", float("nan"), ValueError, "sigma must be finite"),
    ],
)
def test_selection_rejects_malformed_identity_or_sigma_state(
    uuids: object,
    sigma: object,
    error: type[Exception],
    message: str,
) -> None:
    """Reject ambiguous model-call identity and timestep metadata."""

    plan = _scheduled_plan()
    conditioning_uuids = (
        [plan.positive.base_context.entries[0].uuid] if uuids == "valid" else uuids
    )
    with pytest.raises(error, match=message):
        REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
            plan,
            cond_or_uncond=[0],
            conditioning_uuids=conditioning_uuids,
            sigma=sigma,  # type: ignore[arg-type]
        )


def _scheduled_plan() -> ProcessedRegionalAttentionPlan:
    """Build symmetric branches with adjacent and overlapping entry schedules."""

    return _plan(
        positive_base=(
            _entry(0, 1.0, start=100.0, end=50.0),
            _entry(1, 2.0, start=50.0, end=0.0),
        ),
        negative_base=(
            _entry(0, -1.0, start=100.0, end=50.0),
            _entry(1, -2.0, start=50.0, end=0.0),
        ),
        positive_region=_scheduled_region(positive=True),
        negative_region=_scheduled_region(positive=False),
    )


def _gap_plan(*, negative_region: bool = True) -> ProcessedRegionalAttentionPlan:
    """Build branches whose authored regional entries share an inactive gap."""

    gap_entries = (
        _entry(0, 2.0, start=100.0, end=75.0),
        _entry(1, 3.0, start=25.0, end=0.0),
    )
    return _plan(
        positive_base=(_entry(0, 1.0),),
        negative_base=(_entry(0, -1.0),),
        positive_region=gap_entries,
        negative_region=(
            tuple(
                _entry(
                    entry.entry_index,
                    -2.0,
                    start=entry.schedule.timestep_start,
                    end=entry.schedule.timestep_end,
                )
                for entry in gap_entries
            )
            if negative_region
            else None
        ),
    )


def _plan(
    *,
    positive_base: tuple[ProcessedRegionalAttentionEntry, ...],
    negative_base: tuple[ProcessedRegionalAttentionEntry, ...],
    positive_region: tuple[ProcessedRegionalAttentionEntry, ...],
    negative_region: tuple[ProcessedRegionalAttentionEntry, ...] | None,
) -> ProcessedRegionalAttentionPlan:
    """Build one-region processed plan from explicit entry banks."""

    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            ProcessedRegionalAttentionContext(0, None, positive_base),
            (ProcessedRegionalAttentionContext(1, 0, positive_region),),
        ),
        ProcessedRegionalAttentionBranch(
            ProcessedRegionalAttentionContext(0, None, negative_base),
            (
                ()
                if negative_region is None
                else (ProcessedRegionalAttentionContext(1, 0, negative_region),)
            ),
        ),
        RegionalMaskBank(torch.ones((1, 2, 2)), torch.ones((1, 2, 2)), 2, 2),
        RegionalLoraPlan(()),
    )


def _scheduled_region(
    *,
    positive: bool,
) -> tuple[ProcessedRegionalAttentionEntry, ...]:
    """Build one recognizable scheduled region for either branch."""

    sign = 1.0 if positive else -1.0
    return (
        _entry(0, 10.0 * sign, start=100.0, end=50.0, strength=0.1),
        _entry(1, 20.0 * sign, start=50.0, end=0.0, strength=0.2),
        _entry(2, 30.0 * sign, start=75.0, end=25.0, strength=0.3),
        _entry(3, 40.0 * sign, start=50.0, end=50.0, strength=0.4),
    )


def _entry(
    entry_index: int,
    value: float,
    *,
    start: float | None = None,
    end: float | None = None,
    strength: float = 1.0,
) -> ProcessedRegionalAttentionEntry:
    """Build one processed entry with explicit converted schedule boundaries."""

    return ProcessedRegionalAttentionEntry(
        entry_index,
        uuid4(),
        ConditioningScheduleRange(None, None, start, end),
        torch.full((1, 2, 3), value),
        strength,
    )


def _active_base_entry(
    branch: ProcessedRegionalAttentionBranch,
    *,
    sigma: float,
) -> ProcessedRegionalAttentionEntry:
    """Return the single active base entry away from the adjacent boundary."""

    if sigma >= 50.0:
        return branch.base_context.entries[0]
    return branch.base_context.entries[1]
