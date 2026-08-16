# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute exact active, regional-base, and global-base UNet lanes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn

from .standard_unet_variant_graph_attention import (
    StandardUnetVariantGraphAttentionResolver,
)
from .standard_unet_variant_invocation import StandardUnetVariantInvocation
from .standard_unet_variant_lane_plan import StandardUnetVariantLanePlan
from .standard_unet_variant_topology import StandardUnetRegionalVariant


@runtime_checkable
class StandardUnetVariantResolver(Protocol):
    """Expose persistent schedule-specific diffusion shell resolution."""

    def resolve(
        self,
        variant: StandardUnetRegionalVariant,
        schedule_multipliers: tuple[float, ...],
    ) -> nn.Module:
        """Return one resident exact shell."""


@dataclass(frozen=True, slots=True)
class StandardUnetVariantLaneOutputs:
    """Retain ordered regional outputs and an optional global base."""

    region_indices: tuple[int, ...]
    regional_outputs: tuple[torch.Tensor, ...]
    global_base_output: torch.Tensor | None

    def __post_init__(self) -> None:
        """Require one aligned regional output per canonical region index."""

        if len(self.region_indices) != len(self.regional_outputs):
            raise ValueError("Standard UNet lane outputs lost regional alignment.")
        if self.region_indices != tuple(sorted(set(self.region_indices))):
            raise ValueError("Standard UNet lane outputs must be unique and sorted.")


class StandardUnetVariantLaneExecutor:
    """Invoke exact weight states under their selected native conditioning."""

    def __init__(
        self,
        *,
        base_diffusion: nn.Module,
        variant_resolver: StandardUnetVariantResolver,
        graph_attention: StandardUnetVariantGraphAttentionResolver,
    ) -> None:
        """Retain diffusion resolution and phase-owned graph inputs."""

        if not isinstance(base_diffusion, nn.Module):
            raise TypeError("Standard UNet lane execution requires a base module.")
        if not isinstance(variant_resolver, StandardUnetVariantResolver):
            raise TypeError("Standard UNet lane execution requires a resolver.")
        if not isinstance(
            graph_attention,
            StandardUnetVariantGraphAttentionResolver,
        ):
            raise TypeError("Standard UNet lane execution requires graph attention.")
        self._base_diffusion = base_diffusion
        self._variant_resolver = variant_resolver
        self._graph_attention = graph_attention

    def execute(
        self,
        *,
        active_variants: tuple[StandardUnetRegionalVariant, ...],
        schedule_multipliers: tuple[float, ...],
        plan: StandardUnetVariantLanePlan,
        invocation: StandardUnetVariantInvocation,
    ) -> StandardUnetVariantLaneOutputs:
        """Run each required weight/context lane and the optional global base."""

        if not active_variants:
            raise ValueError("Standard UNet lane execution requires active variants.")
        active_by_region = {
            variant.region_index: variant for variant in active_variants
        }
        expected_active = tuple(sorted(active_by_region))
        planned_active = tuple(
            region
            for region in plan.output_region_indices
            if region not in plan.regional_base_indices
        )
        if expected_active != planned_active:
            raise ValueError("Standard UNet lane plan lost active variant ownership.")
        regional_outputs = tuple(
            self._regional_output(
                region_index,
                active_by_region.get(region_index),
                schedule_multipliers,
                invocation,
            )
            for region_index in plan.output_region_indices
        )
        global_base = (
            self.global_base(invocation) if plan.requires_global_base else None
        )
        return StandardUnetVariantLaneOutputs(
            plan.output_region_indices,
            regional_outputs,
            global_base,
        )

    def global_base(self, invocation: StandardUnetVariantInvocation) -> torch.Tensor:
        """Run one packed Attention Couple base for unauthored canvas coverage."""

        attention = self._graph_attention.global_base(invocation)
        return self._run(
            self._base_diffusion,
            invocation,
            context=attention.context,
            transformer_options=attention.transformer_options,
        )

    def _regional_output(
        self,
        region_index: int,
        variant: StandardUnetRegionalVariant | None,
        schedule_multipliers: tuple[float, ...],
        invocation: StandardUnetVariantInvocation,
    ) -> torch.Tensor:
        """Run one active shell or native zero-adapter regional lane."""

        diffusion = (
            self._base_diffusion
            if variant is None
            else self._variant_resolver.resolve(variant, schedule_multipliers)
        )
        attention = self._graph_attention.regional(region_index, invocation)
        return self._run(
            diffusion,
            invocation,
            context=attention.context,
            transformer_options=attention.transformer_options,
        )

    @staticmethod
    def _run(
        diffusion: nn.Module,
        invocation: StandardUnetVariantInvocation,
        *,
        context: torch.Tensor,
        transformer_options: dict[str, object],
    ) -> torch.Tensor:
        """Execute one complete graph with isolated mutable transformer metadata."""

        forward = getattr(diffusion, "_forward", None)
        if not callable(forward):
            raise TypeError("Standard UNet variant graph must expose _forward.")
        args, kwargs = invocation.graph_call(
            context=context,
            transformer_options=transformer_options,
        )
        output = cast(Callable[..., object], forward)(*args, **kwargs)
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError("Standard UNet variant graph must return BCHW output.")
        return output
