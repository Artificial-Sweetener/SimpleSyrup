"""Verify exact task-local reuse of repeated-adapter combined multipliers."""

from __future__ import annotations

from dataclasses import replace

import torch
from regional_lora_test_values import (
    single_region_query_masks,
    single_target_execution,
)

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
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
from simple_syrup.runtime.regional_lora.anima_lora_weights import (
    AnimaLoraWeightResolver,
)
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)


def test_combined_multiplier_reuses_exact_scope_and_separates_strengths() -> None:
    """Cache the ordered sum while retaining schedule and mask-scope identity."""

    first = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    second = AnimaRegionalLoraAdapterExecution(
        replace(first.adapter_plan, composition_index=1),
        first.admission,
        first.attention,
        first.model_lineage,
        first.cache,
    )
    group = AnimaRegionalLoraComposition((first, second)).groups_for_target(
        first.admission.targets[0].adapter.target
    )[0]
    spatial = AnimaLoraSpatialInvocationContext()
    resolver = AnimaLoraWeightResolver(
        cross_attention_context=AnimaCrossAttentionInvocationContext(),
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=spatial,
    )
    inputs = torch.ones((2, 2, 2))
    masks = single_region_query_masks()
    first_resolution = RegionalLoraScheduleResolution(
        (1.0, 1.0),
        (0.25, 0.75),
    )
    second_resolution = RegionalLoraScheduleResolution(
        (1.0, 1.0),
        (0.5, 0.5),
    )

    with spatial.activate(AnimaLoraSpatialInvocation(masks)):
        combined = resolver.combined_multiplier(group, inputs, first_resolution)
        reused = resolver.combined_multiplier(group, inputs, first_resolution)
        changed = resolver.combined_multiplier(group, inputs, second_resolution)

    assert combined is reused
    assert changed is not combined
    assert combined is not None
    expected = masks.flattened[0].clone()
    expected[1] = 0.0
    torch.testing.assert_close(combined, expected)
    assert changed is not None
    torch.testing.assert_close(changed, expected)

    replacement_masks = single_region_query_masks()
    with spatial.activate(AnimaLoraSpatialInvocation(replacement_masks)):
        new_scope = resolver.combined_multiplier(group, inputs, first_resolution)

    assert new_scope is not combined
    assert new_scope is not None
    replacement_expected = replacement_masks.flattened[0].clone()
    replacement_expected[1] = 0.0
    torch.testing.assert_close(new_scope, replacement_expected)
