# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove one shared Anima query-activity plan owns all regional consumers."""

from __future__ import annotations

import torch
from regional_attention_test_values import single_entry_regions

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_branch_batch import (
    AnimaRegionalBranchInvocation,
)
from simple_syrup.runtime.regional_lora.anima_query_activity import (
    AnimaRegionalQueryActivityContext,
)
from simple_syrup.runtime.regional_lora.anima_query_mask_context import (
    AnimaQueryMaskContext,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import AnimaQueryMaskBatch


class _FixedQueryMaskContext(AnimaQueryMaskContext):
    """Return one exact mixed-activity mask batch and record resolutions."""

    def __init__(self, masks: AnimaQueryMaskBatch) -> None:
        """Retain the immutable test batch and initialize its call count."""

        super().__init__()
        self._masks = masks
        self.calls = 0

    def resolve(
        self,
        execution: AnimaRegionalAttentionExecution,
        geometry: AnimaActivationGeometry,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AnimaQueryMaskBatch:
        """Record the validated contract and return its typed mask batch."""

        del execution, geometry
        self.calls += 1
        return AnimaQueryMaskBatch(self._masks.masks.to(device=device, dtype=dtype))


def test_activity_is_shared_and_invalidated_by_dynamic_context_identity() -> None:
    """Reuse one plan until a model-call context authority changes."""

    masks = _mixed_masks()
    query_masks = _FixedQueryMaskContext(masks)
    activity_context = AnimaRegionalQueryActivityContext(query_masks)
    execution = _execution()
    geometry = _geometry()

    first = activity_context.resolve(
        execution,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    repeated = activity_context.resolve(
        execution,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert repeated is first
    assert query_masks.calls == 1
    assert first.attention_branches.invocation == AnimaRegionalBranchInvocation(
        4,
        (0, 1, 2, 3, 0, 2, 1, 2),
        (None, None, None, None, 0, 0, 1, 1),
    )
    assert first.adaln_branches.invocation == AnimaRegionalBranchInvocation(
        4,
        (2, 3, 0, 2, 1, 2),
        (None, None, 0, 0, 1, 1),
    )

    dynamic_contexts = _contexts()
    with execution.execution_context.activate(dynamic_contexts):
        changed = activity_context.resolve(
            execution,
            geometry,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert changed is not first
    assert query_masks.calls == 2


def test_contextual_global_keeps_one_pixel_region_as_fractional_activity() -> None:
    """Preserve sub-token area through attention and LoRA branch pruning."""

    context = torch.zeros((1, 1, 1))
    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        context,
        single_entry_regions((context.clone(),)),
    )
    masks = torch.zeros((1, 8, 8), dtype=torch.float32)
    masks[0, 3, 5] = 1.0
    execution = AnimaRegionalAttentionExecution(
        contexts,
        RegionalMaskBank(masks.clone(), masks, 8, 8),
        (1.0,),
    )
    activity = AnimaRegionalQueryActivityContext().resolve(
        execution,
        _contextual_global_geometry(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        activity.masks.masks,
        torch.tensor([[[[[1.0 / 64.0]]]]]),
    )
    assert float(activity.attention_weights.regions[0, 0, 0]) > 0.0
    assert 0 in activity.attention_branches.invocation.region_indices
    assert 0 in activity.adaln_branches.invocation.region_indices


def _mixed_masks() -> AnimaQueryMaskBatch:
    """Return solid A, solid B, overlap, and uncovered source rows."""

    return AnimaQueryMaskBatch(
        torch.tensor(
            [
                [[[[1.0]]], [[[0.0]]], [[[0.75]]], [[[0.0]]]],
                [[[[0.0]]], [[[1.0]]], [[[0.75]]], [[[0.0]]]],
            ]
        )
    )


def _contexts() -> BatchedRegionalAttentionContexts:
    """Return one four-row context batch with two regional entries."""

    context = torch.zeros((4, 1, 1))
    return BatchedRegionalAttentionContexts(
        4,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 4),),
        context,
        single_entry_regions((context.clone(), context.clone())),
    )


def _execution() -> AnimaRegionalAttentionExecution:
    """Bind the exact context and two-region mask authorities."""

    canonical_masks = torch.ones((2, 1, 1))
    return AnimaRegionalAttentionExecution(
        _contexts(),
        RegionalMaskBank(canonical_masks, canonical_masks.clone(), 1, 1),
        (1.0, 1.0),
    )


def _geometry() -> AnimaActivationGeometry:
    """Return one four-row, one-query activation geometry."""

    return AnimaActivationGeometry(4, 1, 1, 1, 1, 1, 1, 1, 1, None)


def _contextual_global_geometry() -> AnimaActivationGeometry:
    """Reduce an eight-by-eight full-source view to one query token."""

    layout = SpatialBatchLayout(
        8,
        8,
        (
            SpatialView(
                SpatialViewKind.CONTEXTUAL_GLOBAL,
                0,
                0,
                8,
                8,
                4,
                4,
            ),
        ),
        input_batch_size=1,
    )
    return AnimaActivationGeometry(1, 1, 4, 4, 1, 4, 1, 1, 1, layout)
