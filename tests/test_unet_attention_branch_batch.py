"""Verify compact standard-UNet attention branch packing and restoration."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.attention_coupling.unet_branch_batch import (
    UNET_BASE_ATTENTION_BRANCH,
    UnetAttentionBranchBatchBuilder,
    UnetAttentionBranchKey,
)


def test_unet_branch_batch_packs_supported_rows_and_restores_source_order() -> None:
    """Gather only supported rows and scatter each branch to the source batch."""

    regional_key = UnetAttentionBranchKey(0, 0)
    batch = UnetAttentionBranchBatchBuilder().build(
        source_batch_size=3,
        supports=(
            (UNET_BASE_ATTENTION_BRANCH, torch.tensor([True, False, True])),
            (regional_key, torch.tensor([False, True, True])),
        ),
    )
    source = torch.tensor([[1.0], [2.0], [3.0]])
    region = torch.tensor([[10.0], [20.0], [30.0]])

    assert batch.packed_batch_size == 4
    assert torch.equal(
        batch.pack_source(source), torch.tensor([[1.0], [3.0], [2.0], [3.0]])
    )
    assert torch.equal(
        batch.pack_branch_values(
            {
                UNET_BASE_ATTENTION_BRANCH: source,
                regional_key: region,
            }
        ),
        torch.tensor([[1.0], [3.0], [20.0], [30.0]]),
    )
    restored = batch.restore(torch.tensor([[100.0], [300.0], [2000.0], [3000.0]]))
    assert torch.equal(
        restored[UNET_BASE_ATTENTION_BRANCH],
        torch.tensor([[100.0], [0.0], [300.0]]),
    )
    assert torch.equal(
        restored[regional_key],
        torch.tensor([[0.0], [2000.0], [3000.0]]),
    )


def test_unet_branch_batch_prunes_empty_segments_without_reordering() -> None:
    """Retain canonical nonempty branches while dropping unsupported work."""

    batch = UnetAttentionBranchBatchBuilder().build(
        source_batch_size=2,
        supports=(
            (UNET_BASE_ATTENTION_BRANCH, torch.tensor([True, True])),
            (UnetAttentionBranchKey(0, 0), torch.tensor([False, False])),
            (UnetAttentionBranchKey(1, 0), torch.tensor([False, True])),
        ),
    )

    assert tuple(segment.key for segment in batch.segments) == (
        UNET_BASE_ATTENTION_BRANCH,
        UnetAttentionBranchKey(1, 0),
    )
    assert batch.packed_batch_size == 3


def test_unet_branch_batch_rejects_noncanonical_or_empty_support() -> None:
    """Fail before packing when branch order or total support is invalid."""

    builder = UnetAttentionBranchBatchBuilder()
    with pytest.raises(ValueError, match="canonical"):
        builder.build(
            source_batch_size=1,
            supports=(
                (UnetAttentionBranchKey(0, 0), torch.tensor([True])),
                (UNET_BASE_ATTENTION_BRANCH, torch.tensor([True])),
            ),
        )
    with pytest.raises(ValueError, match="cannot all be empty"):
        builder.build(
            source_batch_size=1,
            supports=((UNET_BASE_ATTENTION_BRANCH, torch.tensor([False])),),
        )
