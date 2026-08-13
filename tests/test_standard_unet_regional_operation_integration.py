# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove composed standard-UNet regional operations across spatial modes."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any
from uuid import UUID

import comfy.model_patcher
import pytest
import torch
from comfy.patcher_extension import WrapperExecutor, WrappersMP
from comfy.weight_adapter.lora import LoRAAdapter
from torch import nn

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.attention_coupling.unet import (
    StandardUnetAttentionBackend,
)
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
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
from simple_syrup.runtime.regional_lora.resolved_operation_translator import (
    COMFY_RESOLVED_OPERATION_TRANSLATOR,
)
from simple_syrup.runtime.regional_lora.standard_unet_operation_preparation import (
    StandardUnetOperationAdmission,
)
from simple_syrup.runtime.regional_lora.target_binder import (
    REGIONAL_LORA_TARGET_BINDER,
)
from simple_syrup.runtime.regional_lora.target_binding import (
    BoundRegionalLoraSpatialCapability,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.runtime.spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


@dataclass(frozen=True, slots=True)
class _Mode:
    """Describe one exact standard-UNet operation call layout."""

    mode_id: str
    input_shape: tuple[int, int, int, int]
    layout: SpatialBatchLayout | None


class _OperationDiffusion(nn.Module):
    """Execute one spatial Linear target and record nested denoiser calls."""

    def __init__(self) -> None:
        """Install one identity target and empty call journal."""

        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.linear.weight = nn.Parameter(torch.eye(2), requires_grad=False)
        self.calls = 0

    def forward(
        self,
        model_input: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: object,
        control: object,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Run one B/S/C projection over the current exact activation grid."""

        del timestep, context, y, control
        self.calls += 1
        transformer_options["activations_shape"] = list(model_input.shape)
        tokens = model_input.permute(0, 2, 3, 1).reshape(
            model_input.shape[0],
            -1,
            model_input.shape[1],
        )
        projected_value = self.linear(tokens)
        if not isinstance(projected_value, torch.Tensor):
            raise TypeError("Standard-UNet integration target must return a tensor.")
        projected = projected_value
        return projected.reshape(
            model_input.shape[0],
            model_input.shape[2],
            model_input.shape[3],
            model_input.shape[1],
        ).permute(0, 3, 1, 2)


@pytest.mark.parametrize("mode_id", ["full", "tiled", "contextual"])
@pytest.mark.parametrize("adapter_count", [1, 2, 3])
@pytest.mark.parametrize("region_count", [1, 2])
def test_spatial_modes_keep_one_denoiser_call_for_any_plan_size(
    mode_id: str,
    adapter_count: int,
    region_count: int,
) -> None:
    """Compose regional math without multiplying denoiser trajectories."""

    mode = _mode(mode_id)
    source, diffusion = _patcher()
    plan = _lora_plan(adapter_count, region_count)
    admission = _admission(source, plan)
    state = _state(plan, region_count)
    built = StandardUnetAttentionBackend().derive(
        model=source,
        state=state,
        admission=admission,
    )
    derived = built.model
    assert isinstance(derived, comfy.model_patcher.ModelPatcher)
    original = diffusion.linear
    model_input = torch.arange(
        prod(mode.input_shape),
        dtype=torch.float32,
    ).reshape(mode.input_shape)
    context = torch.zeros((mode.input_shape[0], 2, 2))
    options = _transformer_options(mode, adapter_count)
    expected_scale = 1.0 + sum(adapter.model_strength for adapter in plan.adapters)

    derived.patch_model(load_weights=False)
    try:
        output = WrapperExecutor.new_class_executor(
            diffusion.forward,
            diffusion,
            derived.get_all_wrappers(WrappersMP.DIFFUSION_MODEL),
        ).execute(
            model_input,
            torch.ones(mode.input_shape[0]),
            context,
            None,
            None,
            options,
        )
    finally:
        derived.unpatch_model(unpatch_weights=False)

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(
        output,
        model_input * expected_scale,
        rtol=1e-6,
        atol=2e-6,
    )
    assert diffusion.calls == 1
    assert diffusion.linear is original
    assert source.object_patches == {}
    assert admission.cache is not None
    derived.detach(unpatch_all=True)
    assert admission.cache.size == 0


def _full_mode() -> _Mode:
    """Return one complete 4x2 latent call."""

    return _Mode("full", (1, 2, 2, 4), None)


def _mode(mode_id: str) -> _Mode:
    """Return one declared spatial mode without implicit fallbacks."""

    modes = {
        "full": _full_mode,
        "tiled": _tiled_mode,
        "contextual": _contextual_mode,
    }
    try:
        return modes[mode_id]()
    except KeyError as error:
        raise ValueError(f"Unknown test spatial mode: {mode_id!r}.") from error


def _tiled_mode() -> _Mode:
    """Return two view-major 2x2 tiles over one 4x2 canvas."""

    layout = SpatialBatchLayout(
        4,
        2,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 2, 2, 2, 2),
            SpatialView(SpatialViewKind.TILE, 2, 0, 2, 2, 2, 2),
        ),
        1,
    )
    return _Mode("tiled", (2, 2, 2, 2), layout)


def _contextual_mode() -> _Mode:
    """Return one downscaled full-source Contextual global call."""

    layout = SpatialBatchLayout(
        4,
        2,
        (
            SpatialView(
                SpatialViewKind.CONTEXTUAL_GLOBAL,
                0,
                0,
                4,
                2,
                2,
                2,
            ),
        ),
        1,
    )
    return _Mode("contextual", (1, 2, 2, 2), layout)


def _patcher() -> tuple[comfy.model_patcher.ModelPatcher, _OperationDiffusion]:
    """Return one real CPU patcher around the exact test diffusion target."""

    diffusion = _OperationDiffusion()
    root = nn.Module()
    root.diffusion_model = diffusion
    device = torch.device("cpu")
    return (
        comfy.model_patcher.ModelPatcher(root, device, device),
        diffusion,
    )


def _lora_plan(adapter_count: int, region_count: int) -> RegionalLoraPlan:
    """Return distinct ordered regional adapter uses at one shared target."""

    return RegionalLoraPlan(
        tuple(
            RegionalLoraAdapterPlan(
                RegionalLoraAdapterIdentity(f"adapter-{index}.safetensors"),
                index,
                index % region_count,
                RegionalLoraBranch.POSITIVE,
                0.1 * (index + 1),
                (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
            )
            for index in range(adapter_count)
        )
    )


def _admission(
    source: comfy.model_patcher.ModelPatcher,
    plan: RegionalLoraPlan,
) -> StandardUnetOperationAdmission:
    """Build complete real binder evidence for every declared adapter use."""

    resolved_adapters = tuple(
        ComfyRegionalAdapterResolution(
            adapter,
            RegionalLoraHostPayload(False, None, {}, None),
            (
                ComfyNormalizedAdapterTarget(
                    ComfyAdapterTargetPath("diffusion_model.linear.weight", None),
                    LoRAAdapter(
                        {"up", "down"},
                        (torch.eye(2), torch.eye(2), None, None, None, None),
                    ),
                    "LoRAAdapter",
                    ("down", "up"),
                    True,
                ),
            ),
            (),
            (),
        )
        for adapter in plan.adapters
    )
    resolution = ComfyRegionalLoraResolution(resolved_adapters, ())
    operations = COMFY_RESOLVED_OPERATION_TRANSLATOR.translate(resolution)
    binding = REGIONAL_LORA_TARGET_BINDER.bind(
        source=source,
        candidate=source,
        resolution=resolution,
        operations=operations,
        linear_spatial_capabilities={
            "diffusion_model.linear.weight": (
                BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS
            )
        },
    )
    adaptation = RegionalLoraPlanAdaptation(
        plan,
        tuple(
            RegionalLoraHostPayload(False, None, {}, None) for _adapter in plan.adapters
        ),
    )
    return StandardUnetOperationAdmission(
        adaptation,
        binding,
        {"diffusion_model.linear": BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS},
        RegionalLoraExecutionCache(),
    )


def _state(
    plan: RegionalLoraPlan,
    region_count: int,
) -> StandardUnetAttentionState:
    """Return all-one regional masks and a matching processed LoRA plan."""

    masks = torch.ones((region_count, 2, 4))
    bank = RegionalMaskBank(masks, masks.clone(), 4, 2)
    regional_contexts = tuple(
        _context(region_index + 1, region_index) for region_index in range(region_count)
    )
    processed = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(_context(0, None), regional_contexts),
        ProcessedRegionalAttentionBranch(_context(0, None), regional_contexts),
        bank,
        plan,
    )
    return StandardUnetAttentionState(
        processed,
        (1.0,) * region_count,
        RegionalAttentionDiagnosticsBuilder(bank, backend="test.standard-unet"),
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active compact context with deterministic identity."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                UUID(int=conditioning_index + 1),
                ConditioningScheduleRange(None, None, None, None),
                torch.zeros((1, 2, 2)),
                1.0,
            ),
        ),
    )


def _transformer_options(mode: _Mode, adapter_count: int) -> dict[str, Any]:
    """Return exact CFG, schedule, and optional spatial-layout call metadata."""

    options: dict[str, Any] = {
        "cond_or_uncond": [0] * mode.input_shape[0],
        "sigmas": torch.ones(mode.input_shape[0]),
        "sample_sigmas": torch.tensor([1.0, 0.0]),
    }
    if mode.layout is not None:
        options[SIMPLE_SYRUP_TRANSFORMER_NAMESPACE] = {
            SPATIAL_BATCH_LAYOUT_KEY: mode.layout
        }
    assert adapter_count > 0
    return options
