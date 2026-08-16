# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize graph-attention delegation by standard-UNet variant lanes."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from simple_syrup.domain.regional_lora_plan import RegionalLoraScheduleBoundary
from simple_syrup.runtime.regional_lora.standard_unet_variant_graph_attention import (
    StandardUnetVariantGraphAttention,
    StandardUnetVariantGraphAttentionResolver,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_invocation import (
    StandardUnetVariantInvocation,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_lane_execution import (
    StandardUnetVariantLaneExecutor,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_lane_plan import (
    StandardUnetVariantLanePlan,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_topology import (
    StandardUnetRegionalVariant,
    StandardUnetVariantAdapter,
)


class _Diffusion(nn.Module):
    """Capture exact graph context and options while returning a constant."""

    def __init__(self, value: float) -> None:
        """Retain the output value and graph observations."""

        super().__init__()
        self.value = value
        self.contexts: list[torch.Tensor] = []
        self.options: list[dict[str, object]] = []

    def _forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Return one aligned prediction after recording graph inputs."""

        del kwargs
        self.contexts.append(cast(torch.Tensor, args[2]))
        self.options.append(cast(dict[str, object], args[5]))
        return torch.full_like(cast(torch.Tensor, args[0]), self.value)


class _Resolver:
    """Return the one declared active shell."""

    def __init__(self, active: _Diffusion) -> None:
        """Retain the active shell."""

        self._active = active

    def resolve(
        self,
        variant: StandardUnetRegionalVariant,
        schedule_multipliers: tuple[float, ...],
    ) -> nn.Module:
        """Return the active shell for the canonical left variant."""

        assert variant.region_index == 0
        assert schedule_multipliers == (1.0,)
        return self._active


class _GraphAttention(StandardUnetVariantGraphAttentionResolver):
    """Publish distinct prepared inputs and record delegation."""

    def __init__(self) -> None:
        """Create graph markers without production collaborators."""

        self.regional_indices: list[int] = []
        self.global_calls = 0

    def regional(
        self,
        region_index: int,
        invocation: StandardUnetVariantInvocation,
    ) -> StandardUnetVariantGraphAttention:
        """Return one region-marked context and options dictionary."""

        del invocation
        self.regional_indices.append(region_index)
        return StandardUnetVariantGraphAttention(
            torch.full((1, 1, 1), float(region_index + 1)),
            {"lane": region_index},
        )

    def global_base(
        self,
        invocation: StandardUnetVariantInvocation,
    ) -> StandardUnetVariantGraphAttention:
        """Return one globally marked context and options dictionary."""

        del invocation
        self.global_calls += 1
        return StandardUnetVariantGraphAttention(
            torch.full((1, 1, 1), 9.0),
            {"lane": "global"},
        )


def test_regional_lanes_delegate_all_attention_inputs() -> None:
    """Keep active and native-base graphs free of attention policy."""

    base = _Diffusion(5.0)
    active = _Diffusion(1.0)
    attention = _GraphAttention()
    executor = StandardUnetVariantLaneExecutor(
        base_diffusion=base,
        variant_resolver=_Resolver(active),
        graph_attention=attention,
    )

    outputs = executor.execute(
        active_variants=(_variant(),),
        schedule_multipliers=(1.0,),
        plan=StandardUnetVariantLanePlan((0, 1), (1,), False),
        invocation=_invocation(),
    )

    assert attention.regional_indices == [0, 1]
    assert attention.global_calls == 0
    assert active.contexts[0].item() == 1.0
    assert base.contexts[0].item() == 2.0
    assert active.options == [{"lane": 0}]
    assert base.options == [{"lane": 1}]
    assert outputs.global_base_output is None


def test_global_base_delegates_attention_inputs() -> None:
    """Keep uncovered-canvas global context and callbacks in the focused owner."""

    base = _Diffusion(5.0)
    active = _Diffusion(1.0)
    attention = _GraphAttention()
    executor = StandardUnetVariantLaneExecutor(
        base_diffusion=base,
        variant_resolver=_Resolver(active),
        graph_attention=attention,
    )

    outputs = executor.execute(
        active_variants=(_variant(),),
        schedule_multipliers=(1.0,),
        plan=StandardUnetVariantLanePlan((0,), (), True),
        invocation=_invocation(),
    )

    assert attention.regional_indices == [0]
    assert attention.global_calls == 1
    assert base.contexts[0].item() == 9.0
    assert base.options == [{"lane": "global"}]
    assert outputs.global_base_output is not None


def _variant() -> StandardUnetRegionalVariant:
    """Return one execution-only left adapter variant."""

    schedule = (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),)
    return StandardUnetRegionalVariant(
        0,
        (StandardUnetVariantAdapter(0, 0, 1.0, schedule, ()),),
    )


def _invocation() -> StandardUnetVariantInvocation:
    """Return one normalized graph invocation."""

    return StandardUnetVariantInvocation.bind(
        (torch.zeros((1, 4, 1, 2)), torch.ones(1)),
        {
            "context": torch.zeros((1, 1, 1)),
            "control": None,
            "transformer_options": {},
        },
    )
