# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove Anima AdaLN evaluates only spatially supported branch rows."""

from __future__ import annotations

import torch
from regional_attention_test_values import single_entry_regions
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_attention_weights import (
    RegionalAttentionWeightingPolicy,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.anima_adaln import (
    AnimaRegionalAdalnEvaluator,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_branch_batch import (
    ANIMA_BASE_BRANCH_KEY,
    ANIMA_REGIONAL_BRANCH_BATCH_BUILDER,
    AnimaRegionalBranchInvocation,
    AnimaRegionalBranchKey,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_query_activity import (
    AnimaRegionalQueryActivity,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import AnimaQueryMaskBatch


class _BranchAwareProjection(nn.Module):
    """Return recognizable values for every compact branch invocation."""

    def __init__(self, context: AnimaLoraBranchInvocationContext) -> None:
        """Retain the branch authority and initialize the call record."""

        super().__init__()
        self._context = context
        self.calls: list[torch.Tensor] = []
        self.invocations: list[AnimaRegionalBranchInvocation] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add a branch-specific offset and expand into one triplet."""

        self.calls.append(inputs)
        invocation = self._context.current_or_none()
        if invocation is None:
            raise RuntimeError("Test projection requires an active branch invocation.")
        self.invocations.append(invocation)
        offsets = inputs.new_tensor(
            [
                10.0 if region_index is None else 20.0 + 10.0 * region_index
                for region_index in invocation.region_indices
            ]
        ).reshape(-1, 1, 1)
        return (inputs + offsets).expand(-1, -1, 3)


def test_adaln_prunes_unsupported_rows_and_matches_full_branch_blending() -> None:
    """Compact base and regions once while retaining exact overlap output."""

    masks = AnimaQueryMaskBatch(
        torch.tensor(
            [
                [[[[1.0]]], [[[0.0]]], [[[0.75]]], [[[0.0]]]],
                [[[[0.0]]], [[[1.0]]], [[[0.75]]], [[[0.0]]]],
            ]
        )
    )
    activity = _activity(masks)
    branch_context = AnimaLoraBranchInvocationContext()
    projections = (
        _BranchAwareProjection(branch_context),
        _BranchAwareProjection(branch_context),
        _BranchAwareProjection(branch_context),
    )
    modules = (
        nn.Sequential(nn.Identity(), nn.Identity(), projections[0]),
        nn.Sequential(nn.Identity(), nn.Identity(), projections[1]),
        nn.Sequential(nn.Identity(), nn.Identity(), projections[2]),
    )
    embedding = torch.arange(1.0, 5.0).reshape(4, 1, 1)
    built_in = torch.zeros((4, 1, 3))
    evaluator = AnimaRegionalAdalnEvaluator(_execution(), branch_context)

    outputs = evaluator.evaluate_all(modules, embedding, built_in, activity)

    expected_invocation = AnimaRegionalBranchInvocation(
        4,
        (2, 3, 0, 2, 1, 2),
        (None, None, 0, 0, 1, 1),
    )
    for projection in projections:
        assert len(projection.calls) == 1
        assert projection.calls[0].shape == (6, 1, 1)
        assert projection.invocations == [expected_invocation]
    expected = torch.tensor([21.0, 32.0, 35.5, 14.0]).reshape(4, 1, 1, 1, 1)
    for shift, scale, gate in outputs:
        torch.testing.assert_close(shift, expected)
        torch.testing.assert_close(scale, expected)
        torch.testing.assert_close(gate, expected)


def _activity(masks: AnimaQueryMaskBatch) -> AnimaRegionalQueryActivity:
    """Build the exact prompt and AdaLN compact plans for the fixed masks."""

    weights = RegionalAttentionWeightingPolicy().weights(
        masks.flattened,
        region_strengths=(1.0, 1.0),
    )
    attention = ANIMA_REGIONAL_BRANCH_BATCH_BUILDER.build(
        source_batch_size=4,
        supports=(
            (ANIMA_BASE_BRANCH_KEY, torch.tensor([False, False, False, True])),
            (AnimaRegionalBranchKey(0, 0), torch.tensor([True, False, True, False])),
            (AnimaRegionalBranchKey(1, 0), torch.tensor([False, True, True, False])),
        ),
    )
    adaln = ANIMA_REGIONAL_BRANCH_BATCH_BUILDER.build(
        source_batch_size=4,
        supports=(
            (ANIMA_BASE_BRANCH_KEY, torch.tensor([False, False, True, True])),
            (AnimaRegionalBranchKey(0, 0), torch.tensor([True, False, True, False])),
            (AnimaRegionalBranchKey(1, 0), torch.tensor([False, True, True, False])),
        ),
    )
    return AnimaRegionalQueryActivity(masks, weights, attention, adaln)


def _execution() -> AnimaRegionalAttentionExecution:
    """Return the matching two-region, four-row attention authority."""

    context = torch.zeros((4, 1, 1))
    contexts = BatchedRegionalAttentionContexts(
        4,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 4),),
        context,
        single_entry_regions((context.clone(), context.clone())),
    )
    canonical_masks = torch.ones((2, 1, 1))
    return AnimaRegionalAttentionExecution(
        contexts,
        RegionalMaskBank(canonical_masks, canonical_masks.clone(), 1, 1),
        (1.0, 1.0),
    )
