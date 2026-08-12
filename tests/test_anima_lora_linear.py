# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove every Anima target family receives exact scope-aware LoRA deltas."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from anima_branch_test_values import uniform_branch_invocation
from regional_lora_test_values import (
    single_region_query_masks,
    single_target_execution,
    static_lora_schedule,
)
from torch import nn

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraBranch,
)
from simple_syrup.runtime.regional_lora.anima_branch_batch import (
    AnimaRegionalBranchInvocation,
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
from simple_syrup.runtime.regional_lora.anima_targets import (
    AnimaLoraTarget,
    AnimaLoraTargetFamily,
)

_CROSS_FAMILIES = tuple(
    family for family in AnimaLoraTargetFamily if family.value.startswith("cross_attn")
)
_ADALN_FAMILIES = tuple(
    family for family in AnimaLoraTargetFamily if family.value.startswith("adaln")
)
_SELF_FAMILIES = tuple(
    family for family in AnimaLoraTargetFamily if family.value.startswith("self_attn")
)
_MLP_FAMILIES = tuple(
    family for family in AnimaLoraTargetFamily if family.value.startswith("mlp")
)


class _CountingZeroLinear(nn.Module):
    """Return the original zero projection while counting exact calls."""

    def __init__(self) -> None:
        """Start with no calls."""

        super().__init__()
        self._calls = [0]

    @property
    def calls(self) -> int:
        """Return the shared shell-visible call count."""

        return self._calls[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return an output matching the two-feature synthetic target."""

        self._calls[0] += 1
        return torch.zeros((*inputs.shape[:-1], 2), dtype=inputs.dtype)


def test_linear_backing_module_does_not_leak_a_host_weight_namespace() -> None:
    """Keep the retained installed projection outside PyTorch child discovery."""

    original = nn.Linear(2, 2, bias=False)
    execution = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, _resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        original,
        composition.groups_for_target(execution.admission.targets[0].adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=AnimaLoraSpatialInvocationContext(),
        ),
        schedule_context=schedule,
    )

    assert "original" not in dict(patch.named_modules())
    original_weight = original.weight
    patch.weight = nn.Parameter(torch.full_like(original.weight, 2.0))
    assert original.weight is original_weight


def test_strength_one_branch_multiplier_reuses_validated_broadcast_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache the repeated strength-one branch view shared by linear targets."""

    cross_context = AnimaCrossAttentionInvocationContext()
    resolver = AnimaLoraWeightResolver(
        cross_attention_context=cross_context,
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=AnimaLoraSpatialInvocationContext(),
    )
    execution = single_target_execution(
        AnimaLoraTargetFamily.CROSS_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    target = execution.admission.targets[0]
    inputs = torch.ones((4, 2, 2))
    raw_multiplier = resolver.multiplier
    raw_calls = 0

    def counted_multiplier(
        observed_target: AnimaLoraTarget,
        observed_execution: AnimaRegionalLoraAdapterExecution,
        observed_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Count authoritative raw multiplier resolutions."""

        nonlocal raw_calls
        raw_calls += 1
        return raw_multiplier(observed_target, observed_execution, observed_inputs)

    monkeypatch.setattr(resolver, "multiplier", counted_multiplier)

    with cross_context.activate(uniform_branch_invocation(2, (None, 0))):
        first = resolver.weighted_multiplier(target, execution, inputs, 1.0)
        second = resolver.weighted_multiplier(target, execution, inputs, 1.0)

    assert first is second
    assert raw_calls == 1


def test_weighted_multiplier_applies_adapter_intrinsic_scale() -> None:
    """Apply standard alpha/rank semantics after regional scope resolution."""

    cross_context = AnimaCrossAttentionInvocationContext()
    resolver = AnimaLoraWeightResolver(
        cross_attention_context=cross_context,
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=AnimaLoraSpatialInvocationContext(),
    )
    execution = single_target_execution(
        AnimaLoraTargetFamily.CROSS_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    original = execution.admission.targets[0]
    target = replace(
        original,
        adapter=replace(original.adapter, intrinsic_scale=0.5),
    )
    inputs = torch.ones((4, 2, 2))

    with cross_context.activate(uniform_branch_invocation(2, (None, 0))):
        multiplier = resolver.weighted_multiplier(target, execution, inputs, 1.0)

    assert torch.equal(
        multiplier,
        torch.tensor([[0.0], [0.0], [0.5], [0.0]]),
    )


def test_spatial_multiplier_matches_projection_compute_dtype() -> None:
    """Cast residual-owned masks once for a lower-precision target projection."""

    spatial_context = AnimaLoraSpatialInvocationContext()
    resolver = AnimaLoraWeightResolver(
        cross_attention_context=AnimaCrossAttentionInvocationContext(),
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=spatial_context,
    )
    execution = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    target = execution.admission.targets[0]
    inputs = torch.ones((2, 2, 2), dtype=torch.float16)

    with spatial_context.activate(
        AnimaLoraSpatialInvocation(single_region_query_masks())
    ):
        multiplier = resolver.weighted_multiplier(target, execution, inputs, 1.0)

    assert multiplier.dtype is inputs.dtype
    assert multiplier.device == inputs.device


@pytest.mark.parametrize("family", tuple(AnimaLoraTargetFamily))
def test_each_target_family_adds_exact_delta_in_its_authoritative_scope(
    family: AnimaLoraTargetFamily,
) -> None:
    """Gate the full-rank delta by branch or spatial mask without extra calls."""

    original = _CountingZeroLinear()
    cross_context = AnimaCrossAttentionInvocationContext()
    branch_context = AnimaLoraBranchInvocationContext()
    spatial_context = AnimaLoraSpatialInvocationContext()
    execution = single_target_execution(family, RegionalLoraBranch.POSITIVE)
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        original,
        composition.groups_for_target(execution.admission.targets[0].adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=cross_context,
            branch_context=branch_context,
            spatial_context=spatial_context,
        ),
        schedule_context=schedule,
    )

    if family in _CROSS_FAMILIES:
        inputs = torch.ones((4, 2, 2))
        with (
            schedule.activate(resolution),
            cross_context.activate(uniform_branch_invocation(2, (None, 0))),
        ):
            output = patch(inputs)
        multiplier = torch.tensor([0.0, 0.0, 1.0, 0.0])[:, None, None]
    elif family in _ADALN_FAMILIES:
        inputs = torch.ones((4, 1, 2))
        with (
            schedule.activate(resolution),
            branch_context.activate(uniform_branch_invocation(2, (None, 0))),
        ):
            output = patch(inputs)
        multiplier = torch.tensor([0.0, 0.0, 1.0, 0.0])[:, None, None]
    elif family in _SELF_FAMILIES:
        inputs = torch.ones((2, 2, 2))
        with (
            schedule.activate(resolution),
            spatial_context.activate(
                AnimaLoraSpatialInvocation(single_region_query_masks())
            ),
        ):
            output = patch(inputs)
        multiplier = torch.tensor([[0.25, 0.75], [0.0, 0.0]])[:, :, None]
    else:
        assert family in _MLP_FAMILIES
        inputs = torch.ones((2, 1, 1, 2, 2))
        with (
            schedule.activate(resolution),
            spatial_context.activate(
                AnimaLoraSpatialInvocation(single_region_query_masks())
            ),
        ):
            output = patch(inputs)
        multiplier = torch.tensor([[[[0.25, 0.75]]], [[[0.0, 0.0]]]])[..., None]

    torch.testing.assert_close(output, torch.ones_like(output) * 0.5 * multiplier)
    assert original.calls == 1


def test_negative_adapter_gates_only_negative_chunk_in_regional_branch() -> None:
    """Keep positive and negative adapter application independent."""

    family = AnimaLoraTargetFamily.CROSS_ATTN_Q
    context = AnimaCrossAttentionInvocationContext()
    execution = single_target_execution(family, RegionalLoraBranch.NEGATIVE)
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _CountingZeroLinear(),
        composition.groups_for_target(execution.admission.targets[0].adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=context,
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=AnimaLoraSpatialInvocationContext(),
        ),
        schedule_context=schedule,
    )
    inputs = torch.ones((4, 1, 2))

    with (
        schedule.activate(resolution),
        context.activate(uniform_branch_invocation(2, (None, 0))),
    ):
        output = patch(inputs)

    expected = torch.tensor([0.0, 0.0, 0.0, 0.5])[:, None, None].expand_as(output)
    torch.testing.assert_close(output, expected)


def test_compact_branch_multiplier_preserves_source_cfg_identity() -> None:
    """Gate gathered regional rows by their original positive/negative chunks."""

    context = AnimaCrossAttentionInvocationContext()
    execution = single_target_execution(
        AnimaLoraTargetFamily.CROSS_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    resolver = AnimaLoraWeightResolver(
        cross_attention_context=context,
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=AnimaLoraSpatialInvocationContext(),
    )
    invocation = AnimaRegionalBranchInvocation(
        source_batch_size=2,
        source_batch_indices=(1, 0, 1),
        region_indices=(None, 0, 0),
    )
    inputs = torch.ones((3, 1, 2))

    with context.activate(invocation):
        multiplier = resolver.multiplier(
            execution.admission.targets[0],
            execution,
            inputs,
        )

    torch.testing.assert_close(multiplier, torch.tensor([0.0, 1.0, 0.0]))


def test_regional_adapter_gates_every_active_entry_branch_for_its_region() -> None:
    """Apply one regional LoRA identically across repeated conditioning entries."""

    family = AnimaLoraTargetFamily.CROSS_ATTN_Q
    context = AnimaCrossAttentionInvocationContext()
    execution = single_target_execution(family, RegionalLoraBranch.POSITIVE)
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _CountingZeroLinear(),
        composition.groups_for_target(execution.admission.targets[0].adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=context,
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=AnimaLoraSpatialInvocationContext(),
        ),
        schedule_context=schedule,
    )
    inputs = torch.ones((6, 1, 2))

    with (
        schedule.activate(resolution),
        context.activate(uniform_branch_invocation(2, (None, 0, 0))),
    ):
        output = patch(inputs)

    expected = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.5, 0.0])[:, None, None]
    torch.testing.assert_close(output, expected.expand_as(output))


@pytest.mark.parametrize(
    "family",
    [
        AnimaLoraTargetFamily.CROSS_ATTN_Q,
        AnimaLoraTargetFamily.ADALN_SELF_ATTN_1,
        AnimaLoraTargetFamily.SELF_ATTN_Q,
    ],
)
def test_target_families_fail_closed_without_required_execution_scope(
    family: AnimaLoraTargetFamily,
) -> None:
    """Reject deltas when their branch or spatial authority is unavailable."""

    execution = single_target_execution(family, RegionalLoraBranch.POSITIVE)
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _CountingZeroLinear(),
        composition.groups_for_target(execution.admission.targets[0].adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=AnimaLoraSpatialInvocationContext(),
        ),
        schedule_context=schedule,
    )
    inputs = (
        torch.ones((4, 1, 2))
        if family in (*_CROSS_FAMILIES, *_ADALN_FAMILIES)
        else torch.ones((2, 1, 2))
    )

    with schedule.activate(resolution):
        with pytest.raises(RuntimeError, match="outside|unavailable"):
            patch(inputs)
