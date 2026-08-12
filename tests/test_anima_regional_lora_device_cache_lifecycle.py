"""Prove regional-LoRA device state follows Comfy MODEL detach lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from comfy.patcher_extension import CallbacksMP
from regional_lora_test_values import (
    single_region_query_masks,
    single_target_execution,
    static_lora_schedule,
)

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.runtime.regional_lora.active_support import (
    RegionalLoraActiveSupportResolver,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_device_cache_lifecycle import (
    AnimaRegionalLoraDeviceCacheLifecycle,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocation,
    AnimaLoraSpatialInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_linear_execution import (
    AnimaRegionalLoraCompositionLinearPatch,
)
from simple_syrup.runtime.regional_lora.anima_lora_weights import (
    AnimaLoraWeightResolver,
)
from simple_syrup.runtime.regional_lora.anima_projection_batch import (
    AnimaProjectionBatchRegistry,
    AnimaProjectionBatchRequest,
)
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)

_DETACH_KEY = "simple_syrup.anima_regional_lora_device_cache"


@dataclass
class _Participant:
    """Expose one stable projection request for registry lifecycle checks."""

    request: AnimaProjectionBatchRequest

    def projection_batch_request(
        self,
        inputs: torch.Tensor,
        resolution: RegionalLoraScheduleResolution,
    ) -> AnimaProjectionBatchRequest:
        """Return the configured request for any validated invocation."""

        del inputs, resolution
        return self.request


def test_detach_releases_and_lazily_repopulates_all_owned_weight_state() -> None:
    """Drop prepared tensors on detach without changing later exact execution."""

    execution = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    spatial_context = AnimaLoraSpatialInvocationContext()
    weight_resolver = AnimaLoraWeightResolver(
        cross_attention_context=AnimaCrossAttentionInvocationContext(),
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=spatial_context,
    )
    projection_batches = AnimaProjectionBatchRegistry()
    target_name = execution.admission.targets[0].adapter.target
    patch = AnimaRegionalLoraCompositionLinearPatch(
        torch.nn.Linear(2, 2, bias=False),
        composition.groups_for_target(target_name),
        weight_resolver=weight_resolver,
        schedule_context=schedule,
        projection_batches=projection_batches,
    )
    lifecycle = AnimaRegionalLoraDeviceCacheLifecycle(
        (patch,),
        composition,
        weight_resolver,
        projection_batches,
    )
    model = _patcher()
    lifecycle.mutation().apply(model)
    inputs = torch.ones((2, 2, 2))
    masks = single_region_query_masks()

    with spatial_context.activate(AnimaLoraSpatialInvocation(masks)):
        request = patch.projection_batch_request(inputs, resolution)
    assert request is not None
    first = request.preparation.weights(device=inputs.device, dtype=inputs.dtype)
    assert execution.cache.size == 1

    model.detach()
    model.detach()

    assert execution.cache.size == 0
    second = request.preparation.weights(device=inputs.device, dtype=inputs.dtype)
    assert execution.cache.size == 1
    assert second is not first
    torch.testing.assert_close(second.down, first.down)
    torch.testing.assert_close(second.up, first.up)
    assert model.get_callbacks(CallbacksMP.ON_DETACH, _DETACH_KEY) == [
        lifecycle.release
    ]


def test_projection_and_task_local_resolvers_release_retained_tensors() -> None:
    """Clear pending projection, multiplier, and sparse-support state exactly."""

    execution = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    spatial_context = AnimaLoraSpatialInvocationContext()
    resolver = AnimaLoraWeightResolver(
        cross_attention_context=AnimaCrossAttentionInvocationContext(),
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=spatial_context,
    )
    patch = AnimaRegionalLoraCompositionLinearPatch(
        torch.nn.Linear(2, 2, bias=False),
        composition.groups_for_target(execution.admission.targets[0].adapter.target),
        weight_resolver=resolver,
        schedule_context=schedule,
    )
    inputs = torch.ones((2, 2, 2))
    masks = single_region_query_masks()
    with spatial_context.activate(AnimaLoraSpatialInvocation(masks)):
        request = patch.projection_batch_request(inputs, resolution)
        first_multiplier = resolver.combined_multiplier(
            composition.groups_for_target(
                execution.admission.targets[0].adapter.target
            )[0],
            inputs,
            resolution,
        )
    assert request is not None
    assert first_multiplier is not None
    registry = AnimaProjectionBatchRegistry()
    leader = _Participant(request)
    follower = _Participant(request)
    registry.register("pair", (leader, follower))
    assert registry.resolve_execution(leader, inputs, resolution) is not None
    registry.clear()
    assert registry.resolve_execution(follower, inputs, resolution) is None
    with spatial_context.activate(AnimaLoraSpatialInvocation(masks)):
        resolver.clear()
        second_multiplier = resolver.combined_multiplier(
            composition.groups_for_target(
                execution.admission.targets[0].adapter.target
            )[0],
            inputs,
            resolution,
        )
    assert second_multiplier is not None
    assert second_multiplier is not first_multiplier
    torch.testing.assert_close(second_multiplier, first_multiplier)

    support_resolver = RegionalLoraActiveSupportResolver()
    multiplier = torch.tensor([1.0, 0.0])
    first_support = support_resolver.resolve((multiplier,), leading_shape=(2,))
    support_resolver.clear()
    second_support = support_resolver.resolve((multiplier,), leading_shape=(2,))
    assert first_support is not None
    assert second_support is not None
    assert second_support is not first_support
    torch.testing.assert_close(second_support.indices, first_support.indices)


def _patcher() -> Any:
    """Create a real CPU Comfy MODEL patcher."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(torch.nn.Linear(2, 2), device, device)
