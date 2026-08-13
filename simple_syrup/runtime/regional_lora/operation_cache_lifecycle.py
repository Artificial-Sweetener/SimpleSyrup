# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Release generic regional operation device state at Comfy detach."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from comfy.patcher_extension import CallbacksMP

from ..model_patcher_mutations import ModelKeyedCallbackMutation
from .execution_cache import RegionalLoraExecutionCache

REGIONAL_OPERATION_DETACH_CALLBACK_KEY = "simple_syrup.regional_operation_plans"


class RegionalOperationPlan(Protocol):
    """Expose the device-preparation release contract of one static plan."""

    def clear(self) -> None:
        """Release every locally retained prepared tensor."""


@dataclass(frozen=True, slots=True)
class RegionalOperationCacheLifecycle:
    """Own complete release for one derived generic regional operation model."""

    plans: tuple[RegionalOperationPlan, ...]
    cache: RegionalLoraExecutionCache

    def __post_init__(self) -> None:
        """Require nonempty identity-unique plans and one shared cache owner."""

        if not isinstance(self.plans, tuple) or not self.plans:
            raise ValueError("Regional operation cache plans cannot be empty.")
        if any(not callable(getattr(plan, "clear", None)) for plan in self.plans):
            raise TypeError("Regional operation cache plan must expose clear().")
        if len({id(plan) for plan in self.plans}) != len(self.plans):
            raise ValueError("Regional operation cache plans must be unique.")
        if not isinstance(self.cache, RegionalLoraExecutionCache):
            raise TypeError("Regional operation cache lifecycle requires a cache.")

    def mutation(self) -> ModelKeyedCallbackMutation:
        """Return the single collision-safe Comfy detach registration."""

        return ModelKeyedCallbackMutation(
            CallbacksMP.ON_DETACH,
            REGIONAL_OPERATION_DETACH_CALLBACK_KEY,
            self.release,
        )

    def release(self, model: object, unpatch_all: bool) -> None:
        """Release local and shared prepared tensors on every detach form."""

        del model, unpatch_all
        for plan in self.plans:
            plan.clear()
        self.cache.clear()
