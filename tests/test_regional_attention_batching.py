# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact CFG-chunk and latent-batch regional context alignment."""

from __future__ import annotations

from uuid import uuid4

import pytest
import torch
from comfy.utils import repeat_to_batch_size

from simple_syrup.domain.conditioning_schedule import UNBOUNDED_CONDITIONING_SCHEDULE
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention import (
    RegionalAttentionBranch,
)
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraPlan
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_attention_batching import (
    REGIONAL_ATTENTION_BATCHING_SERVICE,
)


@pytest.mark.parametrize(
    ("selectors", "latent_batch_size", "expected_base"),
    [
        ([0], 1, [1.0]),
        ([0], 3, [1.0, 1.0, 1.0]),
        ([0, 1], 2, [1.0, 1.0, -1.0, -1.0]),
        ([1, 0], 2, [-1.0, -1.0, 1.0, 1.0]),
        ([1, 0, 1], 1, [-1.0, 1.0, -1.0]),
    ],
)
def test_batching_aligns_cfg_modes_and_repeated_chunks_chunk_major(
    selectors: list[int],
    latent_batch_size: int,
    expected_base: list[float],
) -> None:
    """Align CFG=1, ordinary, reversed, and repeated sequences exactly."""

    plan = _plan()
    aligned = REGIONAL_ATTENTION_BATCHING_SERVICE.align(
        plan,
        base_context=_runtime_base(plan, selectors, latent_batch_size),
        cond_or_uncond=selectors,
        conditioning_uuids=_uuids_for_selectors(plan, selectors),
        sigma=0.5,
        latent_batch_size=latent_batch_size,
    )

    assert aligned.base_context[:, 0, 0].tolist() == expected_base
    assert [chunk.chunk_index for chunk in aligned.chunks] == list(
        range(len(selectors))
    )
    assert [chunk.branch for chunk in aligned.chunks] == [
        RegionalAttentionBranch.POSITIVE
        if selector == 0
        else RegionalAttentionBranch.NEGATIVE
        for selector in selectors
    ]
    assert [(chunk.batch_start, chunk.batch_stop) for chunk in aligned.chunks] == [
        (index * latent_batch_size, (index + 1) * latent_batch_size)
        for index in range(len(selectors))
    ]


def test_batching_builds_canonical_regions_with_base_fallback() -> None:
    """Use each branch base where an authored mask has no regional context."""

    plan = _plan()
    aligned = REGIONAL_ATTENTION_BATCHING_SERVICE.align(
        plan,
        base_context=_runtime_base(plan, [0, 1], 2),
        cond_or_uncond=[0, 1],
        conditioning_uuids=_uuids_for_selectors(plan, [0, 1]),
        sigma=0.5,
        latent_batch_size=2,
    )

    assert len(aligned.regions) == 2
    assert aligned.regions[0].entries[0].context[:, 0, 0].tolist() == [
        2.0,
        2.0,
        -2.0,
        -2.0,
    ]
    assert aligned.regions[1].entries[0].context[:, 0, 0].tolist() == [
        3.0,
        3.0,
        -1.0,
        -1.0,
    ]


def test_batching_preserves_all_regional_entries_and_per_sample_strengths() -> None:
    """Align simultaneous regional entries without collapsing their order."""

    plan = _plan()
    positive_region = _multi_entry_context(
        1,
        0,
        ((2.0, 0.25), (6.0, 0.75)),
    )
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            plan.positive.base_context,
            (positive_region, *plan.positive.regional_contexts[1:]),
        ),
        plan.negative,
        plan.mask_bank,
        plan.lora_plan,
    )

    aligned = REGIONAL_ATTENTION_BATCHING_SERVICE.align(
        plan,
        base_context=_runtime_base(plan, [0], 2),
        cond_or_uncond=[0],
        conditioning_uuids=_uuids_for_selectors(plan, [0]),
        sigma=0.5,
        latent_batch_size=2,
    )

    entries = aligned.regions[0].entries
    assert [entry.entry_index for entry in entries] == [0, 1]
    assert [entry.context[:, 0, 0].tolist() for entry in entries] == [
        [2.0, 2.0],
        [6.0, 6.0],
    ]
    assert [entry.strengths for entry in entries] == [(0.25, 0.25), (0.75, 0.75)]


def test_batching_preserves_positive_and_negative_entry_banks_in_chunk_order() -> None:
    """Align independent simultaneous entry banks for both Comfy branches."""

    plan = _plan()
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            plan.positive.base_context,
            (_multi_entry_context(1, 0, ((2.0, 0.25), (6.0, 0.75))),),
        ),
        ProcessedRegionalAttentionBranch(
            plan.negative.base_context,
            (_multi_entry_context(1, 0, ((-2.0, 0.4), (-8.0, 0.6))),),
        ),
        RegionalMaskBank(
            torch.zeros((1, 4, 4)),
            torch.zeros((1, 4, 4)),
            4,
            4,
        ),
        plan.lora_plan,
    )

    aligned = REGIONAL_ATTENTION_BATCHING_SERVICE.align(
        plan,
        base_context=_runtime_base(plan, [1, 0], 1),
        cond_or_uncond=[1, 0],
        conditioning_uuids=_uuids_for_selectors(plan, [1, 0]),
        sigma=0.5,
        latent_batch_size=1,
    )

    entries = aligned.regions[0].entries
    assert [entry.context[:, 0, 0].tolist() for entry in entries] == [
        [-2.0, 2.0],
        [-8.0, 6.0],
    ]
    assert [entry.strengths for entry in entries] == [(0.4, 0.25), (0.6, 0.75)]


def test_batching_uses_comfy_batch_repeat_semantics_without_token_repeat() -> None:
    """Repeat only source batches while preserving the semantic sequence axis."""

    plan = _plan(positive_base_values=(1.0, 2.0))

    aligned = REGIONAL_ATTENTION_BATCHING_SERVICE.align(
        plan,
        base_context=_runtime_base(plan, [0], 3),
        cond_or_uncond=[0],
        conditioning_uuids=_uuids_for_selectors(plan, [0]),
        sigma=0.5,
        latent_batch_size=3,
    )

    assert aligned.base_context[:, 0, 0].tolist() == [1.0, 2.0, 1.0]
    assert aligned.base_context.shape == (3, 2, 3)


@pytest.mark.parametrize(
    ("selectors", "batch_size", "error", "message"),
    [
        ([], 1, ValueError, "at least one Comfy chunk"),
        ([0], 0, ValueError, "must be positive"),
        ([0], True, TypeError, "must be an integer"),
        ([2], 1, ValueError, "observed 2"),
    ],
)
def test_batching_rejects_invalid_chunk_or_batch_inputs(
    selectors: list[int],
    batch_size: int,
    error: type[Exception],
    message: str,
) -> None:
    """Reject ambiguous chunk and latent-batch alignment inputs."""

    with pytest.raises(error, match=message):
        plan = _plan()
        REGIONAL_ATTENTION_BATCHING_SERVICE.align(
            plan,
            base_context=plan.positive.base_context.entries[0].cross_attention,
            cond_or_uncond=selectors,
            conditioning_uuids=_uuids_for_selectors(plan, selectors),
            sigma=0.5,
            latent_batch_size=batch_size,
        )


def test_batching_rejects_cross_branch_sequence_or_dtype_mismatch() -> None:
    """Preflight all possible branches before concatenating partial batches."""

    plan = _plan(negative_dtype=torch.float16)
    with pytest.raises(ValueError, match="dtypes must match"):
        REGIONAL_ATTENTION_BATCHING_SERVICE.align(
            plan,
            base_context=torch.cat(
                (
                    plan.positive.base_context.entries[0].cross_attention,
                    plan.positive.base_context.entries[0].cross_attention,
                )
            ),
            cond_or_uncond=[0, 1],
            conditioning_uuids=_uuids_for_selectors(plan, [0, 1]),
            sigma=0.5,
            latent_batch_size=1,
        )

    plan = _plan(negative_sequence_length=3)
    with pytest.raises(ValueError, match="sequence shapes must match"):
        REGIONAL_ATTENTION_BATCHING_SERVICE.align(
            plan,
            base_context=torch.cat(
                (
                    plan.positive.base_context.entries[0].cross_attention,
                    plan.positive.base_context.entries[0].cross_attention,
                )
            ),
            cond_or_uncond=[0, 1],
            conditioning_uuids=_uuids_for_selectors(plan, [0, 1]),
            sigma=0.5,
            latent_batch_size=1,
        )


def test_batched_value_rejects_noncontiguous_chunk_ranges() -> None:
    """Keep model-batch offsets authoritative and contiguous."""

    context = torch.ones((2, 2, 3))
    with pytest.raises(ValueError, match="contiguous and ordered"):
        BatchedRegionalAttentionContexts(
            1,
            (
                RegionalAttentionChunkBatch(
                    0,
                    RegionalAttentionBranch.POSITIVE,
                    0,
                    1,
                ),
                RegionalAttentionChunkBatch(
                    1,
                    RegionalAttentionBranch.NEGATIVE,
                    2,
                    3,
                ),
            ),
            context,
            (),
        )


def _plan(
    *,
    positive_base_values: tuple[float, ...] = (1.0,),
    negative_dtype: torch.dtype = torch.float32,
    negative_sequence_length: int = 2,
) -> ProcessedRegionalAttentionPlan:
    """Build distinguishable branches with one missing negative region."""

    positive = ProcessedRegionalAttentionBranch(
        _context(0, None, positive_base_values),
        (
            _context(1, 0, (2.0,)),
            _context(2, 1, (3.0,)),
        ),
    )
    negative = ProcessedRegionalAttentionBranch(
        _context(
            0,
            None,
            (-1.0,),
            dtype=negative_dtype,
            sequence_length=negative_sequence_length,
        ),
        (
            _context(
                1,
                0,
                (-2.0,),
                dtype=negative_dtype,
                sequence_length=negative_sequence_length,
            ),
        ),
    )
    planning = torch.zeros((2, 4, 4))
    return ProcessedRegionalAttentionPlan(
        positive,
        negative,
        RegionalMaskBank(planning, torch.zeros_like(planning), 4, 4),
        RegionalLoraPlan(()),
    )


def _runtime_base(
    plan: ProcessedRegionalAttentionPlan,
    selectors: list[int],
    latent_batch_size: int,
) -> torch.Tensor:
    """Return Comfy's chunk-major model-consumed base context fixture."""

    branches = {0: plan.positive, 1: plan.negative}
    return torch.cat(
        tuple(
            repeat_to_batch_size(
                branches[selector].base_context.entries[0].cross_attention,
                latent_batch_size,
            )
            for selector in selectors
        )
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    values: tuple[float, ...],
    *,
    dtype: torch.dtype = torch.float32,
    sequence_length: int = 2,
) -> ProcessedRegionalAttentionContext:
    """Build one BxSxD context with recognizable batch values."""

    tensor = (
        torch.tensor(values, dtype=dtype)
        .reshape(-1, 1, 1)
        .expand(
            -1,
            sequence_length,
            3,
        )
    )
    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (_entry(0, tensor, 1.0),),
    )


def _multi_entry_context(
    conditioning_index: int,
    region_index: int,
    entries: tuple[tuple[float, float], ...],
) -> ProcessedRegionalAttentionContext:
    """Build ordered scalar contexts with explicit Comfy strengths."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        tuple(
            _entry(
                entry_index,
                torch.full((1, 2, 3), value),
                strength,
            )
            for entry_index, (value, strength) in enumerate(entries)
        ),
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
