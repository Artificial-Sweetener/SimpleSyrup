# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Emit common Anima regional diagnostics at the diffusion boundary."""

from __future__ import annotations

import logging

from ...shared.logging import get_logger
from ..diffusion_wrapper_executor import DiffusionWrapperExecutor
from ..model_patcher_mutations import ModelDiffusionWrapperMutation
from .anima_activation_context import AnimaActivationContext
from .anima_attention_diagnostics import AnimaAttentionDiagnosticsBuilder
from .anima_composition_phase_context import AnimaCompositionPhaseContext
from .anima_composition_phase_diagnostics import AnimaCompositionPhaseDiagnostics
from .anima_model_reference import AnimaDiffusionModelReference
from .anima_module_surface import AnimaModuleSurface

ANIMA_ATTENTION_DIAGNOSTICS_WRAPPER_KEY = "simple_syrup.anima_attention_diagnostics"
LOGGER = get_logger("runtime.attention_coupling.anima_diagnostics")


class AnimaAttentionDiagnosticsDiffusionWrapper:
    """Emit one common snapshot around an attention-only Anima call."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        activation_context: AnimaActivationContext,
        builder: AnimaAttentionDiagnosticsBuilder,
        phase_context: AnimaCompositionPhaseContext,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        """Retain verified model, activation, builder, and logging authorities."""

        if not isinstance(surface, AnimaModuleSurface):
            raise TypeError("Anima diagnostic wrapper requires a module surface.")
        if not isinstance(activation_context, AnimaActivationContext):
            raise TypeError("Anima diagnostic wrapper requires activation context.")
        if not isinstance(builder, AnimaAttentionDiagnosticsBuilder):
            raise TypeError("Anima diagnostic wrapper requires an attention builder.")
        if not isinstance(phase_context, AnimaCompositionPhaseContext):
            raise TypeError("Anima diagnostic wrapper requires a phase context.")
        if not isinstance(logger, logging.Logger):
            raise TypeError("Anima diagnostic wrapper requires a logger.")
        self._model = AnimaDiffusionModelReference(surface.diffusion_model)
        self._activation_context = activation_context
        self._builder = builder
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
        if self._logger.isEnabledFor(logging.INFO):
            snapshot = self._builder.build(
                geometry,
                transformer_options=transformer_options,
            )
            fields = snapshot.to_log_fields()
            fields["composition_phase"] = AnimaCompositionPhaseDiagnostics.from_phase(
                self._phase_context.require_current()
            ).to_log_fields()
            self._logger.info(
                "Anima Attention Coupling execution",
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


def anima_attention_diagnostics_wrapper_mutation(
    surface: AnimaModuleSurface,
    activation_context: AnimaActivationContext,
    builder: AnimaAttentionDiagnosticsBuilder,
    phase_context: AnimaCompositionPhaseContext,
) -> ModelDiffusionWrapperMutation:
    """Build the collision-safe attention-only diagnostic mutation."""

    return ModelDiffusionWrapperMutation(
        key=ANIMA_ATTENTION_DIAGNOSTICS_WRAPPER_KEY,
        wrapper=AnimaAttentionDiagnosticsDiffusionWrapper(
            surface,
            activation_context,
            builder,
            phase_context,
        ),
    )
