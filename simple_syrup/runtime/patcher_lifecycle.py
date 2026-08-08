# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own lifecycle-safe derivation of ComfyUI MODEL and CLIP values."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast


class ModelMutation(Protocol):
    """Apply one supported mutation to an already-derived MODEL patcher."""

    def apply(self, model: object) -> None:
        """Apply the mutation through ComfyUI's public patcher API."""


class ClipMutation(Protocol):
    """Apply one supported mutation to an already-derived CLIP value."""

    def apply(self, clip: object) -> None:
        """Apply the mutation through ComfyUI's public CLIP API."""


@dataclass(frozen=True)
class ModelDenoiseMaskMutation:
    """Install a denoise-mask function through the MODEL patcher API."""

    function: Callable[..., object]

    def apply(self, model: object) -> None:
        """Install the configured denoise-mask function."""

        setter = getattr(model, "set_model_denoise_mask_function", None)
        if not callable(setter):
            raise TypeError("MODEL does not support denoise-mask functions.")
        setter(self.function)


@dataclass(frozen=True)
class ModelUnetWrapperMutation:
    """Install a model-function wrapper through the MODEL patcher API."""

    wrapper: Callable[..., object]

    def apply(self, model: object) -> None:
        """Install the configured model-function wrapper."""

        setter = getattr(model, "set_model_unet_function_wrapper", None)
        if not callable(setter):
            raise TypeError("MODEL does not support model-function wrappers.")
        setter(self.wrapper)


@dataclass(frozen=True)
class ModelCalcCondBatchMutation:
    """Install a calc-cond-batch function through the MODEL patcher API."""

    function: Callable[..., object]

    def apply(self, model: object) -> None:
        """Install the configured calc-cond-batch function."""

        setter = getattr(model, "set_model_sampler_calc_cond_batch_function", None)
        if not callable(setter):
            raise TypeError("MODEL does not support calc-cond-batch functions.")
        setter(self.function)


@dataclass(frozen=True)
class ClipLayerMutation:
    """Select the text-encoder layer used by a derived CLIP value."""

    layer_index: int

    def apply(self, clip: object) -> None:
        """Apply the configured CLIP layer index."""

        select_layer = getattr(clip, "clip_layer", None)
        if not callable(select_layer):
            raise TypeError("CLIP does not support layer selection.")
        select_layer(self.layer_index)


@dataclass(frozen=True)
class ClipHookScheduleMutation:
    """Install a native ComfyUI hook schedule on a derived CLIP value."""

    hooks: object
    target: object

    def apply(self, clip: object) -> None:
        """Clone and register hooks through the derived CLIP patcher."""

        patcher = _required_attribute(clip, "patcher", value_name="CLIP")
        clone_hooks = getattr(self.hooks, "clone", None)
        if not callable(clone_hooks):
            raise TypeError("HOOKS does not support lifecycle-safe cloning.")
        register_hooks = getattr(patcher, "register_all_hook_patches", None)
        if not callable(register_hooks):
            raise TypeError("CLIP patcher does not support hook registration.")

        patcher_boundary = cast(Any, patcher)
        clip_boundary = cast(Any, clip)
        patcher_boundary.forced_hooks = clone_hooks()
        clip_boundary.use_clip_schedule = True
        register_hooks(self.hooks, self.target)


PatcherValue = TypeVar("PatcherValue")


class ComfyPatcherLifecycle:
    """Derive Comfy patchers while preserving their source lineage."""

    def derive_model(
        self,
        source: PatcherValue,
        mutations: Iterable[ModelMutation],
        *,
        operation: str,
    ) -> PatcherValue:
        """Clone one MODEL, verify its lineage, and apply all mutations."""

        derived = self._clone(source, operation=operation)
        self._require_direct_parent(source, derived, operation=operation)
        for mutation in mutations:
            mutation.apply(derived)
        return derived

    def derive_clip(
        self,
        source: PatcherValue,
        mutations: Iterable[ClipMutation],
        *,
        operation: str,
        disable_dynamic: bool = False,
    ) -> PatcherValue:
        """Clone one CLIP, verify patcher lineage, and apply all mutations."""

        clone = getattr(source, "clone", None)
        if not callable(clone):
            raise TypeError(f"{operation} requires a cloneable CLIP value.")
        derived = cast(
            PatcherValue,
            clone(disable_dynamic=disable_dynamic) if disable_dynamic else clone(),
        )
        if derived is source:
            raise RuntimeError(f"{operation} returned the source CLIP from clone().")

        source_patcher = _required_attribute(
            source,
            "patcher",
            value_name="source CLIP",
        )
        derived_patcher = _required_attribute(
            derived,
            "patcher",
            value_name="derived CLIP",
        )
        self._require_direct_parent(
            source_patcher,
            derived_patcher,
            operation=operation,
        )
        for mutation in mutations:
            mutation.apply(derived)
        return derived

    def preserve_vae(self, vae: PatcherValue, *, operation: str) -> PatcherValue:
        """Return an unmodified VAE and make the no-derivation contract explicit."""

        if not operation.strip():
            raise ValueError("VAE lifecycle operations require a descriptive name.")
        return vae

    @staticmethod
    def _clone(source: PatcherValue, *, operation: str) -> PatcherValue:
        """Clone one MODEL through its ComfyUI boundary."""

        clone = getattr(source, "clone", None)
        if not callable(clone):
            raise TypeError(f"{operation} requires a cloneable MODEL value.")
        derived = cast(PatcherValue, clone())
        if derived is source:
            raise RuntimeError(f"{operation} returned the source MODEL from clone().")
        return derived

    @staticmethod
    def _require_direct_parent(
        source: object,
        derived: object,
        *,
        operation: str,
    ) -> None:
        """Require Comfy's parent link used by loaded-model cleanup."""

        if getattr(derived, "parent", None) is not source:
            raise RuntimeError(
                f"{operation} produced a derived patcher without its source as parent."
            )


PATCHER_LIFECYCLE = ComfyPatcherLifecycle()


def _required_attribute(value: object, name: str, *, value_name: str) -> object:
    """Return a required dynamic ComfyUI boundary attribute."""

    attribute = getattr(value, name, None)
    if attribute is None:
        raise TypeError(f"{value_name} does not expose {name}.")
    return attribute
