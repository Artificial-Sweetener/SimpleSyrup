# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the production standard-UNet backend through installed transformers."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
import torch
from comfy.ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from comfy.patcher_extension import WrapperExecutor
from torch import nn

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.unet import StandardUnetAttentionBackend
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora.standard_unet_operation_preparation import (
    StandardUnetOperationAdmission,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation


class _ZeroAttention(nn.Module):
    """Observe one retained self-attention call and return zero."""

    def __init__(self) -> None:
        """Initialize the invocation count."""

        super().__init__()
        self.calls = 0

    def forward(
        self,
        query: torch.Tensor,
        *,
        context: torch.Tensor | None,
        value: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Return zero without changing the native block trajectory."""

        del context, value, transformer_options
        self.calls += 1
        return torch.zeros_like(query)


class _ContextAttention(nn.Module):
    """Expose compact branch context values through one attention call."""

    def __init__(self) -> None:
        """Initialize the invocation count."""

        super().__init__()
        self.calls = 0

    def forward(
        self,
        query: torch.Tensor,
        *,
        context: torch.Tensor | None,
        value: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Broadcast one scalar per compact context row over each query."""

        del value, transformer_options
        self.calls += 1
        if context is None:
            raise AssertionError("UNet integration requires cross-attention context.")
        scalar = context.mean(dim=(1, 2), keepdim=True)
        return scalar.expand_as(query)


class _ZeroFeedForward(nn.Module):
    """Observe one retained feed-forward call and return zero."""

    def __init__(self) -> None:
        """Initialize the invocation count."""

        super().__init__()
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return zero without changing the native block trajectory."""

        self.calls += 1
        return torch.zeros_like(value)


class _ResolutionDiffusionModel(nn.Module):
    """Run installed transformers at repeated and reduced resolutions."""

    def __init__(self, context_dimension: int) -> None:
        """Build one observable installed transformer block."""

        super().__init__()
        self.transformer = SpatialTransformer(
            in_channels=32,
            n_heads=1,
            d_head=32,
            depth=1,
            context_dim=context_dimension,
            use_checkpoint=False,
        )
        self.transformer.norm = nn.Identity()
        self.transformer.proj_in = nn.Identity()
        self.transformer.proj_out = nn.Identity()
        block = self.transformer.transformer_blocks[0]
        if not isinstance(block, BasicTransformerBlock):
            raise AssertionError("Installed transformer returned an invalid block.")
        self.self_attention = _ZeroAttention()
        self.cross_attention = _ContextAttention()
        self.feed_forward = _ZeroFeedForward()
        block.norm1 = nn.Identity()
        block.attn1 = self.self_attention
        block.norm2 = nn.Identity()
        block.attn2 = self.cross_attention
        block.norm3 = nn.Identity()
        block.ff = self.feed_forward
        self.outputs: list[torch.Tensor] = []

    def forward(
        self,
        model_input: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        control: object,
        transformer_control: object,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Evaluate two unique query grids across three installed blocks."""

        del timestep, control, transformer_control
        self.outputs.clear()
        for block, transformer_index, height, width in (
            (("input", 4), 2, 4, 6),
            (("middle", 0), 7, 4, 6),
            (("output", 5), 10, 2, 3),
        ):
            options = {
                **transformer_options,
                "original_shape": list(model_input.shape),
                "block": block,
                "transformer_index": transformer_index,
            }
            result = self.transformer(
                torch.zeros(model_input.shape[0], 32, height, width),
                context=context,
                transformer_options=options,
            )
            if not isinstance(result, torch.Tensor):
                raise AssertionError("Installed transformer returned a non-tensor.")
            self.outputs.append(result)
        return self.outputs[-1]


@pytest.mark.parametrize("context_dimension", [768, 2048])
def test_backend_projects_unique_resolutions_once_in_one_native_trajectory(
    context_dimension: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise SD1/SDXL-like contexts through the sole production path."""

    diffusion_model = _ResolutionDiffusionModel(context_dimension)
    source = _patcher(diffusion_model)
    state = _state(context_dimension)
    admission = StandardUnetOperationAdmission(
        RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ()),
        None,
        {},
        None,
    )
    built = StandardUnetAttentionBackend().derive(
        model=source,
        state=state,
        admission=admission,
    )
    derived: Any = built.model
    wrappers = derived.get_all_wrappers("diffusion_model")
    patches = derived.model_options["transformer_options"]["patches"]
    model_input = torch.zeros(1, 4, 8, 12)
    base_context = state.plan.positive.base_context.entries[0].cross_attention
    caplog.set_level(
        "INFO",
        logger="simple_syrup.runtime.attention_coupling.unet_diagnostics",
    )

    output = WrapperExecutor.new_class_executor(
        diffusion_model.forward,
        diffusion_model,
        wrappers,
    ).execute(
        model_input,
        torch.tensor([0.5]),
        base_context,
        None,
        None,
        {
            "cond_or_uncond": [0],
            "sigmas": torch.tensor([0.5]),
            "patches": patches,
        },
    )

    assert output is diffusion_model.outputs[-1]
    assert len(diffusion_model.outputs) == 3
    assert torch.equal(diffusion_model.outputs[0], _expected_high_resolution())
    assert torch.equal(diffusion_model.outputs[1], _expected_high_resolution())
    assert torch.equal(diffusion_model.outputs[2], _expected_low_resolution())
    assert diffusion_model.self_attention.calls == 3
    assert diffusion_model.cross_attention.calls == 3
    assert diffusion_model.feed_forward.calls == 3
    records = [
        record
        for record in caplog.records
        if getattr(record, "operation", None) == "unet_attention_coupling.resolve"
    ]
    assert len(records) == 2
    heights: list[object] = []
    widths: list[object] = []
    denoiser_multipliers: list[object] = []
    for record in records:
        layer = record.__dict__.get("unet_layer")
        diagnostics = record.__dict__.get("regional_diagnostics")
        assert isinstance(layer, dict)
        assert isinstance(diagnostics, dict)
        work = diagnostics.get("estimated_work")
        assert isinstance(work, dict)
        heights.append(layer.get("query_height"))
        widths.append(layer.get("query_width"))
        denoiser_multipliers.append(work.get("denoiser_call_multiplier"))
    assert heights == [4, 2]
    assert widths == [6, 3]
    assert denoiser_multipliers == [1.0, 1.0]
    assert torch.equal(model_input, torch.zeros_like(model_input))
    with pytest.raises(RuntimeError, match="outside"):
        state.execution_context.require_current()
    with pytest.raises(RuntimeError, match="outside"):
        _ = state.resolution_cache.size


def _expected_high_resolution() -> torch.Tensor:
    """Return the exact left/right result at the four-by-six query grid."""

    columns = torch.tensor([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
    return columns.reshape(1, 1, 1, 6).expand(1, 32, 4, 6)


def _expected_low_resolution() -> torch.Tensor:
    """Return the area-preserved result at the two-by-three query grid."""

    columns = torch.tensor([3.0, 2.0, 1.0])
    return columns.reshape(1, 1, 1, 3).expand(1, 32, 2, 3)


def _state(context_dimension: int) -> StandardUnetAttentionState:
    """Return one full-canvas plan with a hard left-half region."""

    masks = torch.zeros(1, 8, 12)
    masks[:, :, :6] = 1.0
    mask_bank = RegionalMaskBank(masks, masks.clone(), 12, 8)
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _context(0, None, 1.0, context_dimension),
            (_context(1, 0, 3.0, context_dimension),),
        ),
        ProcessedRegionalAttentionBranch(
            _context(0, None, -1.0, context_dimension),
            (_context(1, 0, -3.0, context_dimension),),
        ),
        mask_bank,
        EMPTY_REGIONAL_LORA_PLAN,
    )
    return StandardUnetAttentionState(
        plan,
        (1.0,),
        RegionalAttentionDiagnosticsBuilder(
            mask_bank,
            backend="comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel",
        ),
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
    context_dimension: int,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active model-consumed context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.full((1, 3, context_dimension), value),
                1.0,
            ),
        ),
    )


def _patcher(diffusion_model: nn.Module) -> Any:
    """Create one real CPU Comfy ModelPatcher around the test diffusion model."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    base_model = nn.Module()
    base_model.diffusion_model = diffusion_model
    return ModelPatcher(base_model, load_device=device, offload_device=device)
