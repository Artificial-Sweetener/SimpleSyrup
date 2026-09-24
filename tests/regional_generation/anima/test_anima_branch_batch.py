# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove compact Anima branch packing and exact source-batch restoration."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.regional_lora.anima_branch_batch import (
    ANIMA_BASE_BRANCH_KEY,
    AnimaRegionalBranchBatch,
    AnimaRegionalBranchBatchBuilder,
    AnimaRegionalBranchKey,
    AnimaRegionalBranchSegment,
)


def test_batch_packs_only_supported_rows_and_restores_every_branch_key() -> None:
    """Drop zero-support rows without losing source or regional identity."""

    region_zero = AnimaRegionalBranchKey(0, 0)
    region_one = AnimaRegionalBranchKey(1, 0)
    batch = AnimaRegionalBranchBatchBuilder().build(
        source_batch_size=4,
        supports=(
            (ANIMA_BASE_BRANCH_KEY, torch.tensor([True, False, True, False])),
            (region_zero, torch.tensor([False, True, False, False])),
            (region_one, torch.tensor([False, False, True, True])),
        ),
    )
    source = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    branch_values = {
        ANIMA_BASE_BRANCH_KEY: source + 10.0,
        region_zero: source + 20.0,
        region_one: source + 30.0,
    }

    assert batch.pack_source(source)[:, 0].tolist() == [0.0, 2.0, 1.0, 2.0, 3.0]
    assert batch.pack_branch_values(branch_values)[:, 0].tolist() == [
        10.0,
        12.0,
        21.0,
        32.0,
        33.0,
    ]
    assert batch.invocation.source_batch_indices == (0, 2, 1, 2, 3)
    assert batch.invocation.region_indices == (None, None, 0, 1, 1)

    restored = batch.restore(torch.arange(5, dtype=torch.float32).reshape(5, 1))
    assert restored[ANIMA_BASE_BRANCH_KEY][:, 0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert restored[region_zero][:, 0].tolist() == [0.0, 2.0, 0.0, 0.0]
    assert restored[region_one][:, 0].tolist() == [0.0, 0.0, 3.0, 4.0]


def test_builder_prunes_empty_segments_but_rejects_an_empty_call() -> None:
    """Omit unsupported regional work while retaining at least one model row."""

    batch = AnimaRegionalBranchBatchBuilder().build(
        source_batch_size=2,
        supports=(
            (ANIMA_BASE_BRANCH_KEY, torch.tensor([True, True])),
            (AnimaRegionalBranchKey(0, 0), torch.tensor([False, False])),
        ),
    )

    assert tuple(segment.key for segment in batch.segments) == (ANIMA_BASE_BRANCH_KEY,)
    with pytest.raises(ValueError, match="cannot all be empty"):
        AnimaRegionalBranchBatchBuilder().build(
            source_batch_size=2,
            supports=((ANIMA_BASE_BRANCH_KEY, torch.tensor([False, False])),),
        )


def test_batch_rejects_noncanonical_or_out_of_range_segments() -> None:
    """Fail closed on branch order and source-index corruption."""

    regional = AnimaRegionalBranchKey(0, 0)
    with pytest.raises(ValueError, match="unique and canonical"):
        AnimaRegionalBranchBatchBuilder().build(
            source_batch_size=1,
            supports=(
                (regional, torch.tensor([True])),
                (ANIMA_BASE_BRANCH_KEY, torch.tensor([True])),
            ),
        )
    with pytest.raises(ValueError, match="exceeds the source batch"):
        AnimaRegionalBranchBatch(
            1,
            (AnimaRegionalBranchSegment(ANIMA_BASE_BRANCH_KEY, torch.tensor([1])),),
        )
