# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Install generic regional operation patches and detach lifecycle atomically."""

from __future__ import annotations

from dataclasses import dataclass

from comfy.patcher_extension import CallbacksMP

from ..model_object_patch_batch import ModelObjectPatchBatchMutation
from ..model_patcher_mutations import ModelKeyedCallbackMutation
from .operation_cache_lifecycle import (
    REGIONAL_OPERATION_DETACH_CALLBACK_KEY,
    RegionalOperationCacheLifecycle,
)


@dataclass(frozen=True, slots=True)
class RegionalOperationInstallationMutation:
    """Own rollback across exact object patches and detach callback registration."""

    object_patches: ModelObjectPatchBatchMutation
    cache_lifecycle: RegionalOperationCacheLifecycle

    def __post_init__(self) -> None:
        """Require exact focused collaborators before touching patcher state."""

        if not isinstance(self.object_patches, ModelObjectPatchBatchMutation):
            raise TypeError("Regional installation requires an object-patch batch.")
        if not isinstance(self.cache_lifecycle, RegionalOperationCacheLifecycle):
            raise TypeError("Regional installation requires a cache lifecycle.")

    def apply(self, model: object) -> None:
        """Preflight both owners, install both, and rollback either on failure."""

        callback_mutation = self.cache_lifecycle.mutation()
        _preflight_callback(model, callback_mutation)
        object_patches = getattr(model, "object_patches", None)
        callbacks = getattr(model, "callbacks", None)
        if not isinstance(object_patches, dict):
            raise TypeError("MODEL object_patches must be a dictionary.")
        if not isinstance(callbacks, dict):
            raise TypeError("MODEL callbacks must be a dictionary.")
        original_patches = object_patches.copy()
        original_callbacks = _copy_callback_state(callbacks)
        try:
            self.object_patches.apply(model)
            callback_mutation.apply(model)
        except BaseException:
            object_patches.clear()
            object_patches.update(original_patches)
            callbacks.clear()
            callbacks.update(original_callbacks)
            raise


def _preflight_callback(
    model: object,
    mutation: ModelKeyedCallbackMutation,
) -> None:
    """Validate callback collision before installing any object replacement."""

    getter = getattr(model, "get_callbacks", None)
    adder = getattr(model, "add_callback_with_key", None)
    if not callable(getter) or not callable(adder):
        raise TypeError("MODEL must expose keyed callback registration.")
    existing = getter(CallbacksMP.ON_DETACH, REGIONAL_OPERATION_DETACH_CALLBACK_KEY)
    if not isinstance(existing, list):
        raise TypeError("MODEL detach callback state must be a list.")
    if any(not callable(callback) for callback in existing):
        raise TypeError("MODEL detach callbacks must contain only callables.")
    if existing:
        raise ValueError("Regional operation detach callback is already installed.")
    if mutation.callback_type != CallbacksMP.ON_DETACH:
        raise ValueError("Regional installation callback type is invalid.")


def _copy_callback_state(
    callbacks: dict[object, object],
) -> dict[object, object]:
    """Copy Comfy's callback dictionaries and owned callback lists."""

    copied: dict[object, object] = {}
    for event, keyed in callbacks.items():
        if not isinstance(keyed, dict):
            raise TypeError("MODEL callback event state must be a dictionary.")
        copied[event] = {
            key: value.copy() if isinstance(value, list) else value
            for key, value in keyed.items()
        }
    return copied
