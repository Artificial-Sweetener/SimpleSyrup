# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove optimized Anima multi-LoRA composition against ordered references."""

from __future__ import annotations

from uuid import uuid4

import pytest
import torch
from anima_branch_test_values import uniform_branch_invocation
from regional_attention_test_values import single_entry_regions
from regional_lora_test_values import static_lora_schedule
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocation,
    AnimaLoraSpatialInvocationContext,
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_linear_execution import (
    AnimaRegionalLoraCompositionLinearPatch,
)
from simple_syrup.runtime.regional_lora.anima_lora_weights import (
    AnimaLoraWeightResolver,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import AnimaQueryMaskBatch
from simple_syrup.runtime.regional_lora.anima_targets import (
    AnimaLoraAdmission,
    AnimaLoraTarget,
    AnimaLoraTargetFamily,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)


class _CountingZeroLinear(nn.Module):
    """Return a zero original output while counting exact calls."""

    def __init__(self, output_features: int) -> None:
        """Retain the output width and start with no calls."""

        super().__init__()
        self.output_features = output_features
        self._calls = [0]

    @property
    def calls(self) -> int:
        """Return the shared shell-visible call count."""

        return self._calls[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return zeros over every leading input dimension."""

        self._calls[0] += 1
        return torch.zeros(
            (*inputs.shape[:-1], self.output_features),
            device=inputs.device,
            dtype=inputs.dtype,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_batched_pointwise_composition_matches_ordered_unbatched_reference(
    dtype: torch.dtype,
) -> None:
    """Combine repeated uses, batch compatible adapters, and retain exact order."""

    attention = _attention()
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    target_a = _target(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        torch.tensor([[1.0, 0.5], [-0.5, 1.0]]),
        torch.tensor([[0.75, -0.25], [1.0, 0.5]]),
    )
    target_b = _target(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        torch.tensor([[0.25, 1.0], [1.5, -0.5]]),
        torch.tensor([[1.0, 0.25], [-0.75, 0.5]]),
    )
    executions = (
        _execution(0, target_a, "adapter-a", 0, 0.5, attention, model, cache),
        _execution(1, target_a, "adapter-a", 1, 0.25, attention, model, cache),
        _execution(2, target_b, "adapter-b", 0, -0.75, attention, model, cache),
        _execution(3, target_a, "adapter-a", 0, 1.25, attention, model, cache),
    )
    composition = AnimaRegionalLoraComposition(executions)
    groups = composition.groups_for_target(target_a.adapter.target)
    original = _CountingZeroLinear(2)
    cross = AnimaCrossAttentionInvocationContext()
    branch = AnimaLoraBranchInvocationContext()
    spatial = AnimaLoraSpatialInvocationContext()
    schedule, resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        original,
        groups,
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=cross,
            branch_context=branch,
            spatial_context=spatial,
        ),
        schedule_context=schedule,
    )
    inputs = torch.tensor(
        [[[1.0, -0.5], [0.25, 2.0]], [[-1.0, 0.5], [1.5, -0.25]]],
        dtype=dtype,
    )
    masks = _query_masks(dtype)

    with (
        schedule.activate(resolution),
        spatial.activate(AnimaLoraSpatialInvocation(masks)),
    ):
        output = patch(inputs)
        second = patch(inputs)

    expected = _ordered_reference(inputs, executions, masks)
    tolerance = 2e-5 if dtype is torch.float32 else 4e-2
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(second, expected, atol=tolerance, rtol=tolerance)
    assert [len(group.uses) for group in groups] == [2, 1, 1]
    assert original.calls == 2
    assert cache.size == 2
    assert len(patch._rank_batches) == 1
    assert len(patch._rank_batches[0].preparation._prepared) == 1


def test_branch_major_composition_combines_regions_before_b_projection() -> None:
    """Apply one repeated adapter only on its two matching regional branches."""

    attention = _attention()
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    target = _target(
        AnimaLoraTargetFamily.CROSS_ATTN_Q,
        torch.eye(2),
        torch.eye(2),
    )
    executions = (
        _execution(0, target, "adapter-a", 0, 0.5, attention, model, cache),
        _execution(1, target, "adapter-a", 1, -0.25, attention, model, cache),
    )
    composition = AnimaRegionalLoraComposition(executions)
    groups = composition.groups_for_target(target.adapter.target)
    cross = AnimaCrossAttentionInvocationContext()
    schedule, resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _CountingZeroLinear(2),
        groups,
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=cross,
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=AnimaLoraSpatialInvocationContext(),
        ),
        schedule_context=schedule,
    )
    inputs = torch.ones((6, 1, 2))

    with (
        schedule.activate(resolution),
        cross.activate(uniform_branch_invocation(2, (None, 0, 1))),
    ):
        output = patch(inputs)

    expected = torch.tensor([0.0, 0.0, 0.5, 0.0, -0.25, 0.0])[:, None, None]
    torch.testing.assert_close(output, expected.expand_as(output))


def test_composition_prunes_zero_strength_zero_coverage_and_absent_branch() -> None:
    """Omit statically irrelevant target work without weakening active uses."""

    attention = _attention(second_mask_zero=True, negative_chunk=False)
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    target = _target(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        torch.eye(2),
        torch.eye(2),
    )
    executions = (
        _execution(0, target, "adapter-a", 0, 1.0, attention, model, cache),
        _execution(1, target, "adapter-a", 0, 0.0, attention, model, cache),
        _execution(2, target, "adapter-a", 1, 1.0, attention, model, cache),
        _execution(
            3,
            target,
            "adapter-a",
            0,
            1.0,
            attention,
            model,
            cache,
            branch=RegionalLoraBranch.NEGATIVE,
        ),
    )

    groups = AnimaRegionalLoraComposition(executions).groups_for_target(
        target.adapter.target
    )

    assert len(groups) == 1
    assert tuple(
        use.execution.adapter_plan.composition_index for use in groups[0].uses
    ) == (0,)


def test_schedule_zero_prunes_factor_execution_without_repatching() -> None:
    """Skip inactive low-rank factors and reuse the same installed linear patch."""

    attention = _attention()
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    first = _target(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        torch.eye(2),
        torch.eye(2),
    )
    second = _target(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        torch.eye(2) * 2.0,
        torch.eye(2),
    )
    composition = AnimaRegionalLoraComposition(
        (
            _execution(0, first, "first", 0, 1.0, attention, model, cache),
            _execution(1, second, "second", 0, 1.0, attention, model, cache),
        )
    )
    spatial = AnimaLoraSpatialInvocationContext()
    schedule, _static = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _CountingZeroLinear(2),
        composition.groups_for_target(first.adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=spatial,
        ),
        schedule_context=schedule,
    )
    inputs = torch.ones((2, 2, 2))
    masks = AnimaLoraSpatialInvocation(_query_masks(torch.float32))
    inactive = RegionalLoraScheduleResolution((0.0, 0.0), (0.0, 0.0))
    second_only = RegionalLoraScheduleResolution((0.0, 1.0), (0.0, 1.0))

    with schedule.activate(inactive), spatial.activate(masks):
        inactive_output = patch(inputs)
    assert torch.equal(inactive_output, torch.zeros_like(inactive_output))
    assert cache.size == 0
    assert patch._active_batch_preparations == {}

    with schedule.activate(second_only), spatial.activate(masks):
        active_output = patch(inputs)
    assert bool((active_output != 0).any().item())
    assert cache.size == 1
    assert patch._active_batch_preparations == {}


def test_incompatible_ranks_use_separate_batches_without_reordering_addition() -> None:
    """Separate rank groups while retaining the declared group output order."""

    attention = _attention()
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    rank_one = _target(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[1.0], [0.5]]),
    )
    rank_two = _target(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        torch.eye(2),
        torch.eye(2),
    )
    executions = (
        _execution(0, rank_one, "rank-one", 0, 1.0, attention, model, cache),
        _execution(1, rank_two, "rank-two", 0, 1.0, attention, model, cache),
    )
    composition = AnimaRegionalLoraComposition(executions)
    groups = composition.groups_for_target(rank_one.adapter.target)
    spatial = AnimaLoraSpatialInvocationContext()
    schedule, resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _CountingZeroLinear(2),
        groups,
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=spatial,
        ),
        schedule_context=schedule,
    )
    inputs = torch.tensor([[[1.0, 0.5], [-0.25, 2.0]], [[0.5, -1.0], [1.5, 0.25]]])
    masks = _query_masks(torch.float32)

    with (
        schedule.activate(resolution),
        spatial.activate(AnimaLoraSpatialInvocation(masks)),
    ):
        output = patch(inputs)

    assert tuple(batch.indices for batch in patch._rank_batches) == ((0,), (1,))
    torch.testing.assert_close(output, _ordered_reference(inputs, executions, masks))


def test_composition_rejects_invalid_order_and_split_authorities() -> None:
    """Fail before patching when order, attention, or model ownership diverges."""

    attention = _attention()
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    target = _target(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        torch.eye(2),
        torch.eye(2),
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        AnimaRegionalLoraComposition(())
    with pytest.raises(ValueError, match="contiguous declared order"):
        AnimaRegionalLoraComposition(
            (_execution(1, target, "a", 0, 1.0, attention, model, cache),)
        )
    first = _execution(0, target, "a", 0, 1.0, attention, model, cache)
    with pytest.raises(ValueError, match="share one attention"):
        AnimaRegionalLoraComposition(
            (
                first,
                _execution(1, target, "b", 0, 1.0, _attention(), model, cache),
            )
        )
    with pytest.raises(ValueError, match="share one model"):
        AnimaRegionalLoraComposition(
            (
                first,
                _execution(
                    1,
                    target,
                    "b",
                    0,
                    1.0,
                    attention,
                    ModelCloneLineage(uuid4(), uuid4()),
                    cache,
                ),
            )
        )


def _ordered_reference(
    inputs: torch.Tensor,
    executions: tuple[AnimaRegionalLoraAdapterExecution, ...],
    masks: AnimaQueryMaskBatch,
) -> torch.Tensor:
    """Apply one explicit full-rank delta per use in declared order."""

    result = torch.zeros_like(inputs)
    for execution in executions:
        target = execution.admission.targets[0].adapter
        delta = (inputs @ target.down.to(inputs.dtype).T) @ target.up.to(inputs.dtype).T
        mask = masks.flattened[execution.adapter_plan.region_index]
        chunk_gate = torch.tensor([1.0, 0.0], dtype=inputs.dtype)[:, None]
        weight = mask * chunk_gate * float(execution.adapter_plan.model_strength)
        result = result + delta * weight.unsqueeze(-1)
    return result


def _target(
    family: AnimaLoraTargetFamily,
    down: torch.Tensor,
    up: torch.Tensor,
) -> AnimaLoraTarget:
    """Build one admitted synthetic target with caller-selected full rank."""

    name = f"diffusion_model.blocks.0.{family.value}"
    adapter = StandardLoraTarget(
        name,
        down,
        up,
        rank=int(down.shape[0]),
        input_features=int(down.shape[1]),
        output_features=int(up.shape[0]),
    )
    return AnimaLoraTarget(0, family, adapter)


def _execution(
    index: int,
    target: AnimaLoraTarget,
    identity: str,
    region: int,
    strength: float,
    attention: AnimaRegionalAttentionExecution,
    model: object,
    cache: RegionalLoraExecutionCache,
    *,
    branch: RegionalLoraBranch = RegionalLoraBranch.POSITIVE,
) -> AnimaRegionalLoraAdapterExecution:
    """Build one ordered static adapter use around shared authorities."""

    adapter_identity = RegionalLoraAdapterIdentity(identity)
    plan = RegionalLoraAdapterPlan(
        adapter_identity,
        composition_index=index,
        region_index=region,
        branch=branch,
        model_strength=strength,
        schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
    )
    return AnimaRegionalLoraAdapterExecution(
        plan,
        AnimaLoraAdmission((target,)),
        attention,
        ModelCloneLineage.from_model(model),
        cache,
    )


def _attention(
    *,
    second_mask_zero: bool = False,
    negative_chunk: bool = True,
) -> AnimaRegionalAttentionExecution:
    """Build two regions over positive and optional negative aligned chunks."""

    chunk_count = 2 if negative_chunk else 1
    context = torch.zeros((chunk_count, 1, 2))
    chunks = [RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1)]
    if negative_chunk:
        chunks.append(
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.NEGATIVE, 1, 2)
        )
    second = [0.0, 0.0] if second_mask_zero else [0.0, 1.0]
    mask = torch.tensor([[[1.0, 0.0]], [[*second]]])
    return AnimaRegionalAttentionExecution(
        BatchedRegionalAttentionContexts(
            1,
            tuple(chunks),
            context,
            single_entry_regions((context.clone(), context.clone())),
        ),
        RegionalMaskBank(mask.clone(), mask.clone(), 2, 1),
        (1.0, 1.0),
    )


def _query_masks(dtype: torch.dtype) -> AnimaQueryMaskBatch:
    """Return region masks repeated across positive and negative chunks."""

    masks = torch.tensor(
        [
            [[[[1.0, 0.0]]], [[[1.0, 0.0]]]],
            [[[[0.0, 1.0]]], [[[0.0, 1.0]]]],
        ],
        dtype=dtype,
    )
    return AnimaQueryMaskBatch(masks)
