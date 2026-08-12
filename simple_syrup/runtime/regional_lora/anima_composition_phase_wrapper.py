# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve one coordinated Anima composition phase per model call."""

from __future__ import annotations

import torch

from ..diffusion_wrapper_executor import DiffusionWrapperExecutor
from ..model_patcher_mutations import ModelDiffusionWrapperMutation
from ..regional_attention_model_call_values import uniform_model_call_sigma
from .anima_composition_phase import (
    ANIMA_COMPOSITION_PHASE_SCHEDULE,
    AnimaCompositionPhaseSchedule,
)
from .anima_composition_phase_context import AnimaCompositionPhaseContext
from .anima_model_reference import AnimaDiffusionModelReference
from .anima_module_surface import AnimaModuleSurface

ANIMA_COMPOSITION_PHASE_WRAPPER_KEY = "simple_syrup.anima_composition_phase"


class AnimaCompositionPhaseDiffusionWrapper:
    """Publish coordinated scene-building state around one model call."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        context: AnimaCompositionPhaseContext,
        *,
        schedule: AnimaCompositionPhaseSchedule = ANIMA_COMPOSITION_PHASE_SCHEDULE,
    ) -> None:
        """Bind the installed model and focused phase authorities."""

        if not isinstance(surface, AnimaModuleSurface):
            raise TypeError("Anima phase wrapper requires a module surface.")
        if not isinstance(context, AnimaCompositionPhaseContext):
            raise TypeError("Anima phase wrapper requires its task-local context.")
        if not isinstance(schedule, AnimaCompositionPhaseSchedule):
            raise TypeError("Anima phase wrapper requires a phase schedule.")
        self._model = AnimaDiffusionModelReference(surface.diffusion_model)
        self._context = context
        self._schedule = schedule

    def __call__(
        self,
        executor: DiffusionWrapperExecutor,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Resolve exact Comfy sigma metadata and execute under its phase."""

        if not self._model.owns(executor.class_obj):
            raise ValueError("Anima phase wrapper does not own the verified model.")
        options = kwargs.get("transformer_options")
        if not isinstance(options, dict):
            raise TypeError("Anima phase transformer_options must be a dictionary.")
        sample_sigmas = options.get("sample_sigmas")
        current_sigmas = options.get("sigmas")
        if not isinstance(sample_sigmas, torch.Tensor):
            raise TypeError("Anima phase requires tensor sample_sigmas.")
        if not isinstance(current_sigmas, torch.Tensor):
            raise TypeError("Anima phase requires tensor sigmas.")
        phase = self._schedule.resolve(
            sample_sigmas,
            uniform_model_call_sigma(current_sigmas),
        )
        with self._context.activate(phase):
            return executor(*args, **kwargs)


def anima_composition_phase_wrapper_mutation(
    surface: AnimaModuleSurface,
    context: AnimaCompositionPhaseContext,
) -> ModelDiffusionWrapperMutation:
    """Build the collision-safe coordinated phase wrapper mutation."""

    return ModelDiffusionWrapperMutation(
        ANIMA_COMPOSITION_PHASE_WRAPPER_KEY,
        AnimaCompositionPhaseDiffusionWrapper(surface, context),
    )
