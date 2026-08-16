# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own lifecycle-safe derivation of ComfyUI MODEL and CLIP values."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar, cast

from .clip_patcher_model_alignment import align_clip_text_encoder_with_patcher


class ModelMutation(Protocol):
    """Apply one supported mutation to an already-derived MODEL patcher."""

    def apply(self, model: object) -> None:
        """Apply the mutation through ComfyUI's public patcher API."""


class ClipMutation(Protocol):
    """Apply one supported mutation to an already-derived CLIP value."""

    def apply(self, clip: object) -> None:
        """Apply the mutation through ComfyUI's public CLIP API."""


PatcherValue = TypeVar("PatcherValue")


class ComfyPatcherLifecycle:
    """Derive Comfy patchers while preserving their source lineage."""

    def derive_model(
        self,
        source: PatcherValue,
        mutations: Iterable[ModelMutation],
        *,
        operation: str,
        disable_dynamic: bool = False,
    ) -> PatcherValue:
        """Clone one MODEL in the requested host mode and apply all mutations."""

        derived = self._clone(
            source,
            operation=operation,
            disable_dynamic=disable_dynamic,
        )
        self._require_direct_parent(source, derived, operation=operation)
        for mutation in mutations:
            mutation.apply(derived)
        return derived

    def derive_model_with_override(
        self,
        source: PatcherValue,
        model_override_source: object,
        mutations: Iterable[ModelMutation],
        *,
        operation: str,
        disable_dynamic: bool = False,
    ) -> PatcherValue:
        """Clone current request state over another patcher's model allocation."""

        getter = getattr(model_override_source, "get_clone_model_override", None)
        if not callable(getter):
            raise TypeError(f"{operation} requires a model-override source.")
        derived = self._clone(
            source,
            operation=operation,
            disable_dynamic=disable_dynamic,
            model_override=getter(),
        )
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

        source_patcher = self._required_clip_patcher(source, value_name="source CLIP")
        derived_patcher = self._required_clip_patcher(
            derived,
            value_name="derived CLIP",
        )
        self._require_direct_parent(
            source_patcher,
            derived_patcher,
            operation=operation,
        )
        align_clip_text_encoder_with_patcher(derived)
        for mutation in mutations:
            mutation.apply(derived)
        return derived

    def preserve_vae(self, vae: PatcherValue, *, operation: str) -> PatcherValue:
        """Return an unmodified VAE and make the no-derivation contract explicit."""

        if not operation.strip():
            raise ValueError("VAE lifecycle operations require a descriptive name.")
        return vae

    def clone_hooks(
        self,
        source: PatcherValue,
        *,
        operation: str,
    ) -> PatcherValue:
        """Clone one HookGroup-like value through the lifecycle authority."""

        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("Hook lifecycle operations require a description.")
        clone = getattr(source, "clone", None)
        if not callable(clone):
            raise TypeError(f"{operation} requires cloneable hooks.")
        derived = cast(PatcherValue, clone())
        if derived is source:
            raise RuntimeError(f"{operation} returned the source hooks from clone().")
        return derived

    @staticmethod
    def _clone(
        source: PatcherValue,
        *,
        operation: str,
        disable_dynamic: bool,
        model_override: object | None = None,
    ) -> PatcherValue:
        """Clone one MODEL through Comfy's native dynamic-mode boundary."""

        clone = getattr(source, "clone", None)
        if not callable(clone):
            raise TypeError(f"{operation} requires a cloneable MODEL value.")
        if model_override is not None:
            derived = cast(
                PatcherValue,
                clone(
                    disable_dynamic=disable_dynamic,
                    model_override=model_override,
                ),
            )
        else:
            derived = cast(
                PatcherValue,
                clone(disable_dynamic=True) if disable_dynamic else clone(),
            )
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

    @staticmethod
    def _required_clip_patcher(value: object, *, value_name: str) -> object:
        """Return the CLIP patcher required for lineage validation."""

        patcher = getattr(value, "patcher", None)
        if patcher is None:
            raise TypeError(f"{value_name} does not expose patcher.")
        return patcher


PATCHER_LIFECYCLE = ComfyPatcherLifecycle()
