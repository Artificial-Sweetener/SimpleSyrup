# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Release Anima regional-LoRA device caches at Comfy's detach boundary."""

from __future__ import annotations

from dataclasses import dataclass

from comfy.patcher_extension import CallbacksMP

from ..model_patcher_mutations import ModelKeyedCallbackMutation
from .active_support import REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER
from .anima_composition import AnimaRegionalLoraComposition
from .anima_linear_execution import AnimaRegionalLoraCompositionLinearPatch
from .anima_lora_weights import AnimaLoraWeightResolver
from .anima_projection_batch import AnimaProjectionBatchRegistry

_DETACH_CALLBACK_KEY = "simple_syrup.anima_regional_lora_device_cache"


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraDeviceCacheLifecycle:
    """Coordinate release across the exact cache owners of one derived model."""

    patches: tuple[AnimaRegionalLoraCompositionLinearPatch, ...]
    composition: AnimaRegionalLoraComposition
    weight_resolver: AnimaLoraWeightResolver
    projection_batches: AnimaProjectionBatchRegistry

    def __post_init__(self) -> None:
        """Require non-empty, typed, identity-unique release collaborators."""

        self._validate_unique(self.patches, AnimaRegionalLoraCompositionLinearPatch)
        if not isinstance(self.composition, AnimaRegionalLoraComposition):
            raise TypeError("Anima device-cache lifecycle composition is invalid.")
        if not isinstance(self.weight_resolver, AnimaLoraWeightResolver):
            raise TypeError("Anima device-cache lifecycle weight owner is invalid.")
        if not isinstance(self.projection_batches, AnimaProjectionBatchRegistry):
            raise TypeError("Anima device-cache lifecycle projection owner is invalid.")

    def mutation(self) -> ModelKeyedCallbackMutation:
        """Return the single collision-safe Comfy detach registration."""

        return ModelKeyedCallbackMutation(
            CallbacksMP.ON_DETACH,
            _DETACH_CALLBACK_KEY,
            self.release,
        )

    def release(self, model: object, unpatch_all: bool) -> None:
        """Release device state after either complete or clone-switch detach."""

        del model, unpatch_all
        for patch in self.patches:
            patch.clear_device_cache()
        self.projection_batches.clear()
        self.weight_resolver.clear()
        for cache in {
            id(execution.cache): execution.cache
            for execution in self.composition.executions
        }.values():
            cache.clear()
        REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER.clear()

    @staticmethod
    def _validate_unique(values: tuple[object, ...], value_type: type[object]) -> None:
        """Require a non-empty typed tuple without duplicate owners."""

        if not isinstance(values, tuple) or not values:
            raise ValueError("Anima device-cache lifecycle owners cannot be empty.")
        if any(not isinstance(value, value_type) for value in values):
            raise TypeError("Anima device-cache lifecycle owner has an invalid type.")
        if len({id(value) for value in values}) != len(values):
            raise ValueError("Anima device-cache lifecycle owners must be unique.")
