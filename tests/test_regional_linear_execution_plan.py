# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify generic Linear plans adapt exact bound Comfy identities."""

from __future__ import annotations

import comfy.model_patcher
import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter
from torch import nn

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.linear_execution_plan import (
    RegionalLinearExecutionPlanFactory,
)
from simple_syrup.runtime.regional_lora.resolved_operation_translator import (
    ComfyResolvedOperationTranslator,
)
from simple_syrup.runtime.regional_lora.target_binder import RegionalLoraTargetBinder
from simple_syrup.runtime.regional_lora.target_binding import (
    BoundRegionalLoraSpatialCapability,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload


def test_factory_preserves_exact_tensors_scale_and_model_lineage() -> None:
    """Adapt U2/U3/U4 evidence without copying its normalized tensor pair."""

    model = _patcher()
    target = _target()
    resolution = _resolution(target)
    operations = ComfyResolvedOperationTranslator().translate(resolution)
    path = target.path.parameter_key
    binding = RegionalLoraTargetBinder().bind(
        source=model,
        candidate=model,
        resolution=resolution,
        operations=operations,
        linear_spatial_capabilities={
            path: BoundRegionalLoraSpatialCapability.CONSUMER_SPATIALIZED
        },
    )

    plan = RegionalLinearExecutionPlanFactory().build(
        binding.entries,
        model=model,
        cache=RegionalLoraExecutionCache(),
    )

    operation = target.operation
    assert isinstance(operation, LoRAAdapter)
    up, down, *_rest = operation.weights
    assert plan.uses[0].preparation.target.down is down
    assert plan.uses[0].preparation.target.up is up
    assert (
        plan.uses[0].preparation.model_lineage.clone_base_uuid == model.clone_base_uuid
    )
    assert plan.uses[0].preparation.model_lineage.patches_uuid == model.patches_uuid
    assert plan.uses[0].base_strength == pytest.approx(0.8 * (1.0 / 2.0))


def test_factory_rejects_global_only_linear_binding() -> None:
    """Require U5 consumer geometry before admitting generic regional execution."""

    model = _patcher()
    target = _target()
    resolution = _resolution(target)
    operations = ComfyResolvedOperationTranslator().translate(resolution)
    binding = RegionalLoraTargetBinder().bind(
        source=model,
        candidate=model,
        resolution=resolution,
        operations=operations,
    )

    with pytest.raises(ValueError, match="consumer spatialization"):
        RegionalLinearExecutionPlanFactory().build(
            binding.entries,
            model=model,
            cache=RegionalLoraExecutionCache(),
        )


def _patcher() -> comfy.model_patcher.ModelPatcher:
    """Return one real Comfy patcher over a minimal target graph."""

    root = nn.Module()
    root.diffusion_model = nn.Module()
    root.diffusion_model.layer = nn.Linear(4, 6, bias=False)
    return comfy.model_patcher.ModelPatcher(
        root,
        torch.device("cpu"),
        torch.device("cpu"),
    )


def _target() -> ComfyNormalizedAdapterTarget:
    """Return one exact normalized matrix pair with alpha metadata."""

    return ComfyNormalizedAdapterTarget(
        ComfyAdapterTargetPath("diffusion_model.layer.weight", None),
        LoRAAdapter(
            {"up", "down"},
            (
                torch.ones((6, 2)),
                torch.ones((2, 4)),
                1.0,
                None,
                None,
                None,
            ),
        ),
        "LoRAAdapter",
        ("down", "up"),
        True,
    )


def _resolution(target: ComfyNormalizedAdapterTarget) -> ComfyRegionalLoraResolution:
    """Return one complete U2 resolution around the exact target."""

    adapter = RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity("adapter.safetensors"),
        0,
        0,
        RegionalLoraBranch.POSITIVE,
        0.8,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )
    return ComfyRegionalLoraResolution(
        (
            ComfyRegionalAdapterResolution(
                adapter,
                RegionalLoraHostPayload(False, None, {}, None),
                (target,),
                (),
                (),
            ),
        ),
        (),
    )
