# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Release Anima Attention Coupling device caches on Comfy MODEL detach."""

from __future__ import annotations

from dataclasses import dataclass

from comfy.patcher_extension import CallbacksMP

from ..model_patcher_mutations import ModelKeyedCallbackMutation
from .anima_query_activity import AnimaRegionalQueryActivityContext
from .anima_query_mask_context import AnimaQueryMaskContext

_DETACH_CALLBACK_KEY = "simple_syrup.anima_attention_device_cache"


@dataclass(frozen=True, slots=True)
class AnimaAttentionDeviceCacheLifecycle:
    """Coordinate release across one derived attention execution's cache owners."""

    query_activity: AnimaRegionalQueryActivityContext
    query_masks: AnimaQueryMaskContext

    def __post_init__(self) -> None:
        """Require the two focused attention cache authorities."""

        if not isinstance(self.query_activity, AnimaRegionalQueryActivityContext):
            raise TypeError("Anima attention cache lifecycle activity is invalid.")
        if not isinstance(self.query_masks, AnimaQueryMaskContext):
            raise TypeError("Anima attention cache lifecycle masks are invalid.")

    def mutation(self) -> ModelKeyedCallbackMutation:
        """Return the single collision-safe Comfy detach registration."""

        return ModelKeyedCallbackMutation(
            CallbacksMP.ON_DETACH,
            _DETACH_CALLBACK_KEY,
            self.release,
        )

    def release(self, model: object, unpatch_all: bool) -> None:
        """Release projected tensors after complete or clone-switch detach."""

        del model, unpatch_all
        self.query_activity.clear()
        self.query_masks.clear()
