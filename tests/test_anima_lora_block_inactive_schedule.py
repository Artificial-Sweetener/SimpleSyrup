"""Verify inactive regional LoRA schedules bypass only block-owned work."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from regional_lora_test_values import single_target_execution
from torch import nn

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationContext,
)
from simple_syrup.runtime.regional_lora.anima_block_execution import (
    AnimaRegionalLoraBlockPatch,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_schedule_context import (
    AnimaRegionalLoraScheduleContext,
)
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)


class _PassAttention(nn.Module):
    """Count calls while preserving the supplied tensor."""

    def __init__(self) -> None:
        """Start with no calls."""

        super().__init__()
        self.calls = 0

    def forward(
        self, inputs: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        """Return the input unchanged."""

        self.calls += 1
        return inputs


class _RetainedBlock(nn.Module):
    """Expose the installed block surface and observable original execution."""

    def __init__(self) -> None:
        """Create required child modules and an empty call ledger."""

        super().__init__()
        self.layer_norm_self_attn = nn.Identity()
        self.layer_norm_cross_attn = nn.Identity()
        self.layer_norm_mlp = nn.Identity()
        self.self_attn = _PassAttention()
        self.cross_attn = _PassAttention()
        self.mlp = nn.Identity()
        self.adaln_modulation_self_attn = _modulation()
        self.adaln_modulation_cross_attn = _modulation()
        self.adaln_modulation_mlp = _modulation()
        self.calls: list[tuple[object, ...]] = []

    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        context: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Record exact objects and execute the currently mirrored child."""

        self.calls.append((x, embedding, context, kwargs))
        output = self.cross_attn(x, context, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Test attention must return a tensor.")
        return output


def test_inactive_schedule_delegates_to_retained_patched_block() -> None:
    """Skip regional block machinery while retaining patched attention children."""

    execution = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    original = _RetainedBlock()
    schedule = AnimaRegionalLoraScheduleContext()
    patch = AnimaRegionalLoraBlockPatch(
        original,
        execution.attention,
        activation_context=AnimaActivationContext(),
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=AnimaLoraSpatialInvocationContext(),
        schedule_context=schedule,
    )
    mirrored_attention = _PassAttention()
    patch.cross_attn = mirrored_attention
    x = torch.zeros((1, 1, 1, 1, 2))
    embedding = torch.zeros((1, 1, 2))
    context = torch.zeros((1, 1, 2))
    options = {"sentinel": object()}
    resolution = RegionalLoraScheduleResolution((0.0,), (0.0,))

    with schedule.activate(resolution):
        output = patch(x, embedding, context, transformer_options=options)

    assert output is x
    assert mirrored_attention.calls == 1
    assert original.cross_attn.calls == 0
    assert len(original.calls) == 1
    observed = original.calls[0]
    assert observed[0] is x
    assert observed[1] is embedding
    assert observed[2] is context
    observed_kwargs = observed[3]
    assert isinstance(observed_kwargs, dict)
    assert observed_kwargs["transformer_options"] is options


def test_active_schedule_does_not_take_inactive_bypass() -> None:
    """Require the existing activation scope whenever any strength is nonzero."""

    execution = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    schedule = AnimaRegionalLoraScheduleContext()
    patch = AnimaRegionalLoraBlockPatch(
        _RetainedBlock(),
        execution.attention,
        activation_context=AnimaActivationContext(),
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=AnimaLoraSpatialInvocationContext(),
        schedule_context=schedule,
    )

    with schedule.activate(RegionalLoraScheduleResolution((1.0,), (0.5,))):
        with pytest.raises(RuntimeError, match="activation"):
            patch(
                torch.zeros((1, 1, 1, 1, 2)),
                torch.zeros((1, 1, 2)),
                torch.zeros((1, 1, 2)),
            )


def _modulation() -> nn.Sequential:
    """Return the installed three-stage AdaLN module shape."""

    return nn.Sequential(nn.Identity(), nn.Linear(2, 2), nn.Linear(2, 6))
