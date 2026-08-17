# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute persistent standard-UNet variants and compose complete predictions."""

from __future__ import annotations

import torch
from torch import nn

from ...domain.regional_lora_plan import RegionalLoraPlan
from ...domain.regional_mask_bank import RegionalMaskBank
from ..attention_coupling.unet_attention_phase_session import (
    StandardUnetAttentionPhaseSession,
)
from ..regional_attention_model_call_values import uniform_model_call_sigma
from .standard_unet_composition_diagnostics import (
    STANDARD_UNET_COMPOSITION_DIAGNOSTICS_EMITTER,
    StandardUnetCompositionDiagnostic,
    StandardUnetCompositionDiagnosticsEmitter,
)
from .standard_unet_lora_schedule import StandardUnetLoraSchedule
from .standard_unet_variant_base_attention import StandardUnetVariantBaseAttention
from .standard_unet_variant_conditioning import (
    StandardUnetVariantConditioningResolver,
)
from .standard_unet_variant_graph_attention import (
    StandardUnetVariantGraphAttentionResolver,
)
from .standard_unet_variant_invocation import StandardUnetVariantInvocation
from .standard_unet_variant_lane_execution import (
    StandardUnetVariantLaneExecutor,
    StandardUnetVariantResolver,
)
from .standard_unet_variant_lane_plan import (
    STANDARD_UNET_VARIANT_LANE_PLANNER,
    StandardUnetVariantLanePlan,
    StandardUnetVariantLanePlanner,
)
from .standard_unet_variant_masks import (
    STANDARD_UNET_VARIANT_MASK_PROJECTOR,
    StandardUnetVariantMaskProjector,
)
from .standard_unet_variant_output import (
    STANDARD_UNET_VARIANT_OUTPUT_COMPOSER,
    StandardUnetVariantOutputComposer,
)
from .standard_unet_variant_spatial_context import (
    STANDARD_UNET_VARIANT_SPATIAL_CONTEXT,
    StandardUnetVariantSpatialContext,
)
from .standard_unet_variant_topology import (
    StandardUnetRegionalVariant,
    StandardUnetVariantTopology,
)


class StandardUnetVariantExecution:
    """Own one denoiser call's schedule, graph selection, and output composition."""

    def __init__(
        self,
        *,
        base_diffusion: nn.Module,
        plan: RegionalLoraPlan,
        topology: StandardUnetVariantTopology,
        mask_bank: RegionalMaskBank,
        variant_resolver: StandardUnetVariantResolver,
        conditioning: StandardUnetVariantConditioningResolver,
        attention_phase: StandardUnetAttentionPhaseSession,
        base_attention: StandardUnetVariantBaseAttention,
        lane_planner: StandardUnetVariantLanePlanner = (
            STANDARD_UNET_VARIANT_LANE_PLANNER
        ),
        mask_projector: StandardUnetVariantMaskProjector = (
            STANDARD_UNET_VARIANT_MASK_PROJECTOR
        ),
        output_composer: StandardUnetVariantOutputComposer = (
            STANDARD_UNET_VARIANT_OUTPUT_COMPOSER
        ),
        spatial_context: StandardUnetVariantSpatialContext = (
            STANDARD_UNET_VARIANT_SPATIAL_CONTEXT
        ),
        diagnostics: StandardUnetCompositionDiagnosticsEmitter = (
            STANDARD_UNET_COMPOSITION_DIAGNOSTICS_EMITTER
        ),
    ) -> None:
        """Retain focused call authorities and one task-local schedule cursor."""

        if not isinstance(base_diffusion, nn.Module):
            raise TypeError("Standard UNet variant execution requires a base module.")
        if not isinstance(plan, RegionalLoraPlan) or not plan.adapters:
            raise ValueError("Standard UNet variant execution requires adapters.")
        if (
            not isinstance(topology, StandardUnetVariantTopology)
            or not topology.variants
        ):
            raise ValueError("Standard UNet variant execution requires topology.")
        if not isinstance(mask_bank, RegionalMaskBank):
            raise TypeError("Standard UNet variant execution requires a mask bank.")
        if not isinstance(variant_resolver, StandardUnetVariantResolver):
            raise TypeError("Standard UNet variant execution requires a resolver.")
        if not isinstance(conditioning, StandardUnetVariantConditioningResolver):
            raise TypeError(
                "Standard UNet variant execution requires native conditioning."
            )
        if not isinstance(attention_phase, StandardUnetAttentionPhaseSession):
            raise TypeError("Standard UNet variant execution requires phase state.")
        if not isinstance(base_attention, StandardUnetVariantBaseAttention):
            raise TypeError("Standard UNet variant execution requires base attention.")
        if not isinstance(lane_planner, StandardUnetVariantLanePlanner):
            raise TypeError("Standard UNet variant execution requires a lane planner.")
        if not isinstance(spatial_context, StandardUnetVariantSpatialContext):
            raise TypeError("Standard UNet variant execution requires spatial context.")
        self._base_diffusion = base_diffusion
        self._plan = plan
        self._topology = topology
        self._mask_bank = mask_bank
        self._variant_resolver = variant_resolver
        self._attention_phase = attention_phase
        self._mask_projector = mask_projector
        self._output_composer = output_composer
        self._diagnostics = diagnostics
        self._spatial_context = spatial_context
        self._schedule = StandardUnetLoraSchedule(plan)
        self._lane_planner = lane_planner
        self._lane_executor = StandardUnetVariantLaneExecutor(
            base_diffusion=base_diffusion,
            variant_resolver=variant_resolver,
            graph_attention=StandardUnetVariantGraphAttentionResolver(
                conditioning=conditioning,
                base_attention=base_attention,
            ),
        )
        self._lane_plan_cache: dict[
            tuple[int, ...],
            StandardUnetVariantLanePlan,
        ] = {}

    def prime(self) -> None:
        """Construct time-invariant variants before the sampling clock starts."""

        if not self._plan.is_time_invariant:
            return
        multipliers = tuple(
            adapter.schedule[0].strength_multiplier for adapter in self._plan.adapters
        )
        for variant in self._active_variants(multipliers):
            self._variant_resolver.resolve(variant, multipliers)

    def execute(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Run the minimum complete graph set and compose one model prediction."""

        invocation = StandardUnetVariantInvocation.bind(args, kwargs)
        multipliers = self._schedule.resolve(invocation.transformer_options)
        sampling_sigma = uniform_model_call_sigma(
            invocation.transformer_options.get("sigmas")
        )
        self._diagnostics.emit(
            StandardUnetCompositionDiagnostic(
                self._attention_phase.require_current(),
                multipliers,
                sampling_sigma,
                tuple(adapter.schedule for adapter in self._plan.adapters),
                self._spatial_context.modes(invocation.transformer_options),
            )
        )
        active = self._active_variants(multipliers)
        if not active:
            return self._lane_executor.global_base(invocation)
        region_indices = tuple(variant.region_index for variant in active)
        outputs = self._lane_executor.execute(
            active_variants=active,
            schedule_multipliers=multipliers,
            plan=self._lane_plan(region_indices),
            invocation=invocation,
        )
        masks = self._mask_projector.project(
            bank=self._mask_bank,
            region_indices=outputs.region_indices,
            model_input=invocation.model_input,
            transformer_options=invocation.transformer_options,
        )
        return self._output_composer.compose(
            outputs.regional_outputs,
            masks,
            base_output=outputs.global_base_output,
        )

    def clear(self, model: object, unpatch_all: bool) -> None:
        """Release schedule, coverage, and registered variant state on detach."""

        del model, unpatch_all
        self._schedule.clear()
        self._lane_plan_cache.clear()

    def _active_variants(
        self,
        multipliers: tuple[float, ...],
    ) -> tuple[StandardUnetRegionalVariant, ...]:
        """Return variants with at least one nonzero effective adapter."""

        return tuple(
            variant
            for variant in self._topology.variants
            if any(
                adapter.model_strength * multipliers[adapter.composition_index] != 0.0
                for adapter in variant.adapters
            )
        )

    def _lane_plan(
        self,
        region_indices: tuple[int, ...],
    ) -> StandardUnetVariantLanePlan:
        """Cache canonical lane policy by active region set."""

        cached = self._lane_plan_cache.get(region_indices)
        if cached is None:
            cached = self._lane_planner.plan(
                self._mask_bank,
                region_indices,
            )
            self._lane_plan_cache[region_indices] = cached
        return cached
