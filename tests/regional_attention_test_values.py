"""Build explicit regional-attention batch values for focused tests."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
)


def single_entry_regions(
    contexts: tuple[torch.Tensor, ...],
) -> tuple[BatchedRegionalAttentionRegion, ...]:
    """Wrap each aligned region tensor as one unit-strength active entry."""

    return tuple(
        BatchedRegionalAttentionRegion(
            region_index,
            (
                BatchedRegionalAttentionEntry(
                    0,
                    context,
                    (1.0,) * int(context.shape[0]),
                ),
            ),
        )
        for region_index, context in enumerate(contexts)
    )
