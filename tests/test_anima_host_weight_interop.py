"""Prove Comfy host weight state reaches regional Anima linear backing."""

from __future__ import annotations

import comfy.ops
import torch
from regional_lora_test_values import (
    single_region_query_masks,
    single_target_execution,
    static_lora_schedule,
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
)
from simple_syrup.runtime.regional_lora.anima_linear_execution import (
    AnimaRegionalLoraCompositionLinearPatch,
)
from simple_syrup.runtime.regional_lora.anima_lora_weights import (
    AnimaLoraWeightResolver,
)
from simple_syrup.runtime.regional_lora.anima_schedule_context import (
    AnimaRegionalLoraScheduleContext,
)
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)


def test_host_cast_function_composes_with_unchanged_regional_delta() -> None:
    """Apply Comfy's public global weight function before the regional delta."""

    patch, schedule, resolution, spatial, _original = _patch()
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.5], [2.0, -2.0]]])
    with (
        schedule.activate(resolution),
        spatial.activate(AnimaLoraSpatialInvocation(single_region_query_masks())),
    ):
        regional_only = patch(inputs)
        patch.weight_function = [lambda weight: weight * 2.0]
        patch.comfy_cast_weights = True
        global_and_regional = patch(inputs)

    torch.testing.assert_close(global_and_regional - regional_only, inputs)
    assert patch._backing.module.weight_function is patch.weight_function
    assert patch._backing.module.__dict__["comfy_cast_weights"] is True


def test_host_runtime_attributes_and_direct_state_share_backing_authority() -> None:
    """Mirror Comfy dynamic writes, deletes, and direct parameter replacement."""

    patch, _schedule, _resolution, _spatial, original = _patch()
    source_weight = original.weight
    lowvram_function = object()

    patch.weight_lowvram_function = lowvram_function
    patch.weight_comfy_model_dtype = torch.float16
    patch.seed_key = "diffusion_model.blocks.0.self_attn.q_proj"
    patch._v_signature = (1, 2, 3)

    backing = patch._backing.module
    assert backing.weight_lowvram_function is lowvram_function
    assert backing.__dict__["weight_comfy_model_dtype"] is torch.float16
    assert backing.seed_key == "diffusion_model.blocks.0.self_attn.q_proj"
    assert backing._v_signature == (1, 2, 3)

    patch.to(dtype=torch.float64)
    assert patch.weight is backing.weight
    assert patch.weight.dtype is torch.float64
    assert source_weight.dtype is torch.float32

    del patch._v_signature
    assert not hasattr(patch, "_v_signature")
    assert not hasattr(backing, "_v_signature")


def test_host_backing_supplies_absent_bias_registration_for_dynamic_loading() -> None:
    """Expose the bias=None surface required by Comfy's dynamic size estimator."""

    patch, _schedule, _resolution, _spatial, _original = _patch(
        remove_bias_registration=True
    )

    assert patch.bias is None
    assert patch._backing.module.bias is None
    assert "bias" in patch._parameters


def _patch(
    *, remove_bias_registration: bool = False
) -> tuple[
    AnimaRegionalLoraCompositionLinearPatch,
    AnimaRegionalLoraScheduleContext,
    RegionalLoraScheduleResolution,
    AnimaLoraSpatialInvocationContext,
    torch.nn.Module,
]:
    """Build one installed Comfy linear with a regional identity adapter."""

    original = comfy.ops.disable_weight_init.Linear(2, 2, bias=False)
    original.weight = torch.nn.Parameter(torch.eye(2), requires_grad=False)
    if remove_bias_registration:
        del original._parameters["bias"]
    execution = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    spatial = AnimaLoraSpatialInvocationContext()
    patch = AnimaRegionalLoraCompositionLinearPatch(
        original,
        composition.groups_for_target(execution.admission.targets[0].adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=spatial,
        ),
        schedule_context=schedule,
    )
    return patch, schedule, resolution, spatial, original
