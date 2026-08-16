# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare compact standard-UNet execution with an independent reference oracle."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)

_REGION_STRENGTHS = (0.4, 0.4)


@pytest.mark.parametrize(
    ("branches", "latent_batch_size"),
    (
        ((RegionalAttentionBranch.POSITIVE,), 1),
        (
            (
                RegionalAttentionBranch.NEGATIVE,
                RegionalAttentionBranch.POSITIVE,
            ),
            1,
        ),
        (
            (
                RegionalAttentionBranch.POSITIVE,
                RegionalAttentionBranch.NEGATIVE,
            ),
            2,
        ),
    ),
    ids=("cfg-one", "cfg-reversed", "cfg-latent-batch-two"),
)
def test_compact_execution_matches_independent_reference_attention(
    branches: tuple[RegionalAttentionBranch, ...],
    latent_batch_size: int,
) -> None:
    """Match complete branch attention and reconstruction in source row order."""

    contexts = _contexts(branches, latent_batch_size=latent_batch_size)
    masks = _masks(contexts)
    execution = UnetAttn2Execution(contexts, masks, _REGION_STRENGTHS, 1, 4)
    query = (
        torch.arange(
            int(contexts.base_context.shape[0]) * 4 * 3,
            dtype=torch.float64,
        ).reshape(int(contexts.base_context.shape[0]), 4, 3)
        / 17.0
    )

    expanded = execution.expand(query, contexts.base_context, contexts.base_context)
    packed = _attention(expanded.query, expanded.context, expanded.value)
    actual = execution.blend(packed)

    expected = _reference(query, contexts, masks, _REGION_STRENGTHS)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def _reference(
    query: torch.Tensor,
    contexts: BatchedRegionalAttentionContexts,
    masks: torch.Tensor,
    region_strengths: tuple[float, ...],
) -> torch.Tensor:
    """Evaluate each source row independently with explicit reference ownership."""

    rows: list[torch.Tensor] = []
    for chunk in contexts.chunks:
        for row in range(chunk.batch_start, chunk.batch_stop):
            base = _attention(
                query[row : row + 1],
                contexts.base_context[row : row + 1],
                contexts.base_context[row : row + 1],
            )
            if chunk.branch is RegionalAttentionBranch.NEGATIVE:
                rows.append(base)
                continue
            regional = tuple(
                _attention(
                    query[row : row + 1],
                    region.entries[0].context[row : row + 1],
                    region.entries[0].context[row : row + 1],
                )
                for region in contexts.regions
            )
            strengths = masks.new_tensor(region_strengths).reshape(-1, 1)
            regional_weights = masks[:, row].clamp(0.0, 1.0) * strengths
            regional_sum = regional_weights.sum(dim=0)
            base_strength = max(0.0, 1.0 - max(region_strengths))
            base_weight = torch.where(
                regional_sum.ne(0),
                regional_sum.new_full(regional_sum.shape, base_strength),
                regional_sum.new_ones(regional_sum.shape),
            )
            denominator = (base_weight + regional_sum).clamp_min(1e-6)
            numerator = base * base_weight.unsqueeze(-1)
            numerator += sum(
                output * regional_weights[index].unsqueeze(-1)
                for index, output in enumerate(regional)
            )
            rows.append(numerator / denominator.unsqueeze(-1))
    return torch.cat(rows)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run one-head native scaled dot-product attention over B/Q/D tensors."""

    return functional.scaled_dot_product_attention(
        query.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
    ).squeeze(1)


def _contexts(
    branches: tuple[RegionalAttentionBranch, ...],
    *,
    latent_batch_size: int,
) -> BatchedRegionalAttentionContexts:
    """Return aligned base and two distinguishable single-entry region banks."""

    batch = len(branches) * latent_batch_size
    base = torch.arange(batch * 6 * 3, dtype=torch.float64).reshape(batch, 6, 3)
    base = base / 29.0
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
        (
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, base + 2.0, (1.0,) * batch),),
            ),
            BatchedRegionalAttentionRegion(
                1,
                (BatchedRegionalAttentionEntry(0, base - 3.0, (1.0,) * batch),),
            ),
        ),
    )


def _masks(contexts: BatchedRegionalAttentionContexts) -> torch.Tensor:
    """Return partial, overlapping, and uncovered masks with inactive negatives."""

    batch = int(contexts.base_context.shape[0])
    masks = (
        torch.tensor(
            (
                (1.0, 0.75, 0.0, 0.0),
                (0.0, 0.75, 1.0, 0.0),
            ),
            dtype=torch.float64,
        )
        .unsqueeze(1)
        .expand(-1, batch, -1)
        .clone()
    )
    for chunk in contexts.chunks:
        if chunk.branch is RegionalAttentionBranch.NEGATIVE:
            masks[:, chunk.batch_start : chunk.batch_stop] = 0.0
    return masks
