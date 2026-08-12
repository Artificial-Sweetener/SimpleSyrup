# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve regional LoRA schedules before each nested Anima model execution."""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from comfy.patcher_extension import CallbacksMP

from ..diffusion_wrapper_executor import DiffusionWrapperExecutor
from ..model_patcher_mutations import (
    ModelDiffusionWrapperMutation,
    ModelKeyedCallbackMutation,
)
from ..patcher_lifecycle import ModelMutation
from ..regional_attention_model_call_values import uniform_model_call_sigma
from ..regional_lora_schedule_resolution import RegionalLoraScheduleSession
from .anima_composition import AnimaRegionalLoraComposition
from .anima_composition_phase_context import AnimaCompositionPhaseContext
from .anima_model_reference import AnimaDiffusionModelReference
from .anima_module_surface import AnimaModuleSurface
from .anima_schedule_context import AnimaRegionalLoraScheduleContext

ANIMA_REGIONAL_LORA_SCHEDULE_WRAPPER_KEY = "simple_syrup.anima_regional_lora_schedule"
_SCHEDULE_DETACH_CALLBACK_KEY = "simple_syrup.anima_regional_lora_schedule_session"


@dataclass(frozen=True, slots=True)
class _ScheduleSessionSlot:
    """Bind one exact Comfy sample schedule tensor to its mutable session."""

    sample_sigmas: torch.Tensor
    maximum_sigma: float
    session: RegionalLoraScheduleSession
    last_sigma: float | None


class AnimaRegionalLoraScheduleDiffusionWrapper:
    """Resolve and publish scheduled strengths before Anima block execution."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        composition: AnimaRegionalLoraComposition,
        context: AnimaRegionalLoraScheduleContext,
        phase_context: AnimaCompositionPhaseContext,
    ) -> None:
        """Retain verified model, composition, and task-local authorities."""

        if not isinstance(surface, AnimaModuleSurface):
            raise TypeError("Anima schedule wrapper requires a module surface.")
        if not isinstance(composition, AnimaRegionalLoraComposition):
            raise TypeError("Anima schedule wrapper requires a composition.")
        if not isinstance(context, AnimaRegionalLoraScheduleContext):
            raise TypeError("Anima schedule wrapper requires a schedule context.")
        if not isinstance(phase_context, AnimaCompositionPhaseContext):
            raise TypeError("Anima schedule wrapper requires a phase context.")
        self._model = AnimaDiffusionModelReference(surface.diffusion_model)
        self._composition = composition
        self._context = context
        self._phase_context = phase_context
        self._session: ContextVar[_ScheduleSessionSlot | None] = ContextVar(
            "simple_syrup_anima_regional_lora_schedule_session",
            default=None,
        )

    def __call__(
        self,
        executor: DiffusionWrapperExecutor,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Resolve the current sigma and execute under its active adapter state."""

        if not self._model.owns(executor.class_obj):
            raise ValueError(
                "Anima schedule wrapper executor does not own the verified model."
            )
        options = kwargs.get("transformer_options")
        if not isinstance(options, dict):
            raise TypeError("Anima schedule transformer_options must be a dictionary.")
        sample_sigmas = _floating_tensor(options.get("sample_sigmas"), "sample_sigmas")
        current_sigmas = _floating_tensor(options.get("sigmas"), "sigmas")
        current_sigma = uniform_model_call_sigma(current_sigmas)
        slot = self._session.get()
        if slot is None or slot.sample_sigmas is not sample_sigmas:
            maximum_sigma = _maximum_finite_sigma(sample_sigmas)
        else:
            maximum_sigma = slot.maximum_sigma
        if (
            slot is None
            or slot.sample_sigmas is not sample_sigmas
            or (slot.last_sigma is not None and current_sigma > slot.last_sigma)
        ):
            adapters = tuple(
                execution.adapter_plan for execution in self._composition.executions
            )
            slot = _ScheduleSessionSlot(
                sample_sigmas,
                maximum_sigma,
                RegionalLoraScheduleSession(
                    adapters,
                    maximum_sigma=maximum_sigma,
                ),
                None,
            )
        resolution = slot.session.resolve(current_sigma).scaled(
            self._phase_context.require_current().regional_lora_scale
        )
        self._session.set(
            _ScheduleSessionSlot(
                sample_sigmas,
                slot.maximum_sigma,
                slot.session,
                current_sigma,
            )
        )
        with self._context.activate(resolution):
            return executor(*args, **kwargs)

    def clear(self, model: object, unpatch_all: bool) -> None:
        """Release the current task's retained CUDA sampling schedule."""

        del model, unpatch_all
        self._session.set(None)


def _floating_tensor(value: object, name: str) -> torch.Tensor:
    """Require one non-empty floating Comfy sigma tensor without device work."""

    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.numel() < 1
    ):
        raise TypeError(f"Anima regional LoRA {name} must be a floating tensor.")
    return value


def _maximum_finite_sigma(sample_sigmas: torch.Tensor) -> float:
    """Validate one new sample schedule on CPU and return its maximum."""

    values = tuple(
        float(value) for value in sample_sigmas.flatten().detach().cpu().tolist()
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Anima regional LoRA sample_sigmas must be finite.")
    return max(values)


def anima_schedule_wrapper_mutations(
    surface: AnimaModuleSurface,
    composition: AnimaRegionalLoraComposition,
    context: AnimaRegionalLoraScheduleContext,
    phase_context: AnimaCompositionPhaseContext,
) -> tuple[ModelMutation, ModelMutation]:
    """Build collision-safe schedule execution and detach mutations."""

    wrapper = AnimaRegionalLoraScheduleDiffusionWrapper(
        surface,
        composition,
        context,
        phase_context,
    )
    return (
        ModelDiffusionWrapperMutation(
            ANIMA_REGIONAL_LORA_SCHEDULE_WRAPPER_KEY,
            wrapper,
        ),
        ModelKeyedCallbackMutation(
            CallbacksMP.ON_DETACH,
            _SCHEDULE_DETACH_CALLBACK_KEY,
            wrapper.clear,
        ),
    )
