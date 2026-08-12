# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Emit one structured diagnostic record at the Anima diffusion boundary."""

from __future__ import annotations

import logging

from ...shared.logging import get_logger
from ..diffusion_wrapper_executor import DiffusionWrapperExecutor
from ..model_patcher_mutations import ModelDiffusionWrapperMutation
from .anima_activation_context import AnimaActivationContext
from .anima_composition_phase_context import AnimaCompositionPhaseContext
from .anima_composition_phase_diagnostics import AnimaCompositionPhaseDiagnostics
from .anima_diagnostics import AnimaRegionalDiagnosticsBuilder
from .anima_model_reference import AnimaDiffusionModelReference
from .anima_module_surface import AnimaModuleSurface
from .anima_schedule_context import AnimaRegionalLoraScheduleContext

ANIMA_DIAGNOSTICS_WRAPPER_KEY = "simple_syrup.anima_regional_diagnostics"
LOGGER = get_logger("runtime.regional_lora.anima_diagnostics")


class AnimaRegionalDiagnosticsDiffusionWrapper:
    """Build and emit diagnostics once around one regional diffusion call."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        activation_context: AnimaActivationContext,
        builder: AnimaRegionalDiagnosticsBuilder,
        schedule_context: AnimaRegionalLoraScheduleContext,
        phase_context: AnimaCompositionPhaseContext,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        """Retain verified model, activation, builder, and logging authorities."""

        if not isinstance(surface, AnimaModuleSurface):
            raise TypeError("Anima diagnostic wrapper requires a module surface.")
        if not isinstance(activation_context, AnimaActivationContext):
            raise TypeError("Anima diagnostic wrapper requires activation context.")
        if not isinstance(builder, AnimaRegionalDiagnosticsBuilder):
            raise TypeError("Anima diagnostic wrapper requires a diagnostics builder.")
        if not isinstance(schedule_context, AnimaRegionalLoraScheduleContext):
            raise TypeError("Anima diagnostic wrapper requires a schedule context.")
        if not isinstance(phase_context, AnimaCompositionPhaseContext):
            raise TypeError("Anima diagnostic wrapper requires a phase context.")
        if not isinstance(logger, logging.Logger):
            raise TypeError("Anima diagnostic wrapper requires a logger.")
        self._model = AnimaDiffusionModelReference(surface.diffusion_model)
        self._activation_context = activation_context
        self._builder = builder
        self._schedule_context = schedule_context
        self._phase_context = phase_context
        self._logger = logger

    def __call__(
        self,
        executor: DiffusionWrapperExecutor,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Emit one snapshot and forward every model argument unchanged."""

        if not self._model.owns(executor.class_obj):
            raise ValueError(
                "Anima diagnostic wrapper executor does not own the verified model."
            )
        geometry = self._activation_context.require_current()
        transformer_options = kwargs.get("transformer_options")
        resolution = self._schedule_context.require_current()
        if self._logger.isEnabledFor(logging.INFO):
            snapshot = self._builder.build(
                geometry,
                transformer_options=transformer_options,
                schedule_resolution=resolution,
            )
            fields = snapshot.to_log_fields()
            fields["composition_phase"] = AnimaCompositionPhaseDiagnostics.from_phase(
                self._phase_context.require_current()
            ).to_log_fields()
            self._logger.info(
                "Anima regional Attention Coupling execution",
                extra={
                    "operation": "anima_attention_coupling.execute",
                    "regional_diagnostics": fields,
                },
            )
        else:
            self._builder.validate_call(
                geometry,
                transformer_options=transformer_options,
            )
        return executor(*args, **kwargs)


def anima_diagnostics_wrapper_mutation(
    surface: AnimaModuleSurface,
    activation_context: AnimaActivationContext,
    builder: AnimaRegionalDiagnosticsBuilder,
    schedule_context: AnimaRegionalLoraScheduleContext,
    phase_context: AnimaCompositionPhaseContext,
) -> ModelDiffusionWrapperMutation:
    """Build the collision-safe clone-local diagnostics wrapper mutation."""

    return ModelDiffusionWrapperMutation(
        key=ANIMA_DIAGNOSTICS_WRAPPER_KEY,
        wrapper=AnimaRegionalDiagnosticsDiffusionWrapper(
            surface,
            activation_context,
            builder,
            schedule_context,
            phase_context,
        ),
    )
