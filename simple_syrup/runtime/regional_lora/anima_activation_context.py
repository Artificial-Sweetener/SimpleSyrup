# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish isolated per-call Anima activation geometry."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ...domain.spatial_views import SpatialBatchLayout
from ..diffusion_wrapper_executor import DiffusionWrapperExecutor
from ..model_patcher_mutations import ModelDiffusionWrapperMutation
from ..spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)
from .anima_model_reference import AnimaDiffusionModelReference
from .anima_module_surface import AnimaModuleSurface

ANIMA_ACTIVATION_GEOMETRY_KEY = "anima_activation_geometry"
ANIMA_ACTIVATION_WRAPPER_KEY = "simple_syrup.anima_activation_geometry"


@dataclass(frozen=True)
class AnimaActivationGeometry:
    """Describe one active Anima latent and transformer query grid."""

    input_batch_size: int
    activation_time: int
    activation_height: int
    activation_width: int
    patch_temporal: int
    patch_spatial: int
    query_time: int
    query_height: int
    query_width: int
    spatial_layout: SpatialBatchLayout | None

    @property
    def query_token_count(self) -> int:
        """Return the flattened self-attention query length."""

        return self.query_time * self.query_height * self.query_width


class AnimaActivationContext:
    """Own task-local activation state and namespaced option publication."""

    def __init__(self) -> None:
        """Create one context variable with no process-global active geometry."""

        self._current: ContextVar[AnimaActivationGeometry | None] = ContextVar(
            "simple_syrup_anima_activation_geometry",
            default=None,
        )

    def current_or_none(self) -> AnimaActivationGeometry | None:
        """Return the current task-local geometry when inside a model call."""

        return self._current.get()

    def require_current(self) -> AnimaActivationGeometry:
        """Return active geometry or reject execution outside its wrapper scope."""

        geometry = self.current_or_none()
        if geometry is None:
            raise RuntimeError(
                "Anima activation geometry is unavailable outside the "
                "SimpleSyrup diffusion wrapper."
            )
        return geometry

    @contextmanager
    def activate(self, geometry: AnimaActivationGeometry) -> Iterator[None]:
        """Publish one geometry for exactly the nested diffusion execution."""

        token = self._current.set(geometry)
        try:
            yield
        finally:
            self._current.reset(token)

    @staticmethod
    def from_transformer_options(options: object) -> AnimaActivationGeometry:
        """Read the exact namespaced geometry published for one model call."""

        if not isinstance(options, dict):
            raise TypeError("Anima transformer_options must be a dictionary.")
        namespace = options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
        if not isinstance(namespace, dict):
            raise TypeError(
                "Anima transformer_options.simple_syrup must be a dictionary."
            )
        geometry = namespace.get(ANIMA_ACTIVATION_GEOMETRY_KEY)
        if not isinstance(geometry, AnimaActivationGeometry):
            raise TypeError(
                "Anima transformer options require published SimpleSyrup "
                "activation geometry."
            )
        return geometry


class AnimaActivationDiffusionWrapper:
    """Publish validated geometry around one exact installed Anima model call."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        context: AnimaActivationContext,
    ) -> None:
        """Bind the wrapper to one discovered clone-local diffusion model."""

        self._model = AnimaDiffusionModelReference(surface.diffusion_model)
        self._context = context

    def __call__(
        self,
        executor: DiffusionWrapperExecutor,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Publish copied namespaced metadata for the nested model execution."""

        if not self._model.owns(executor.class_obj):
            raise ValueError(
                "Anima activation wrapper executor does not own the discovered "
                "diffusion model."
            )
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError(
                "Anima activation wrapper requires a tensor model input argument."
            )
        forwarded_kwargs = dict(kwargs)
        transformer_options = forwarded_kwargs.get("transformer_options", {})
        if not isinstance(transformer_options, dict):
            raise TypeError("Anima transformer_options must be a dictionary.")
        geometry = self._geometry(args[0], transformer_options)
        forwarded_kwargs["transformer_options"] = self._publish(
            transformer_options,
            geometry,
        )
        with self._context.activate(geometry):
            return executor(*args, **forwarded_kwargs)

    def _geometry(
        self,
        model_input: torch.Tensor,
        transformer_options: dict[object, object],
    ) -> AnimaActivationGeometry:
        """Derive and validate image activation geometry from one Anima input."""

        if model_input.ndim != 5:
            raise ValueError(
                "Anima activation input must have BxCxTxHxW shape; observed "
                f"{tuple(model_input.shape)}."
            )
        batch, _, time, height, width = (int(value) for value in model_input.shape)
        if batch <= 0 or time <= 0 or height <= 0 or width <= 0:
            raise ValueError("Anima activation dimensions must all be positive.")
        if time != 1:
            raise ValueError(
                "Anima regional Attention Coupling currently requires image "
                f"activations with T=1; observed T={time}."
            )
        patch_temporal = self._model.patch_temporal
        patch_spatial = self._model.patch_spatial
        layout = self._spatial_layout(transformer_options)
        if layout is not None:
            self._validate_layout(
                layout,
                activation_batch=batch,
                activation_height=height,
                activation_width=width,
            )
        return AnimaActivationGeometry(
            input_batch_size=batch,
            activation_time=time,
            activation_height=height,
            activation_width=width,
            patch_temporal=patch_temporal,
            patch_spatial=patch_spatial,
            query_time=_ceil_div(time, patch_temporal),
            query_height=_ceil_div(height, patch_spatial),
            query_width=_ceil_div(width, patch_spatial),
            spatial_layout=layout,
        )

    @staticmethod
    def _spatial_layout(
        transformer_options: dict[object, object],
    ) -> SpatialBatchLayout | None:
        """Read the current layout from the existing SimpleSyrup namespace."""

        namespace = transformer_options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
        if namespace is None:
            return None
        if not isinstance(namespace, dict):
            raise TypeError("transformer_options.simple_syrup must be a dictionary.")
        layout = namespace.get(SPATIAL_BATCH_LAYOUT_KEY)
        if layout is None:
            return None
        if not isinstance(layout, SpatialBatchLayout):
            raise TypeError(
                "transformer_options.simple_syrup.spatial_batch_layout must be "
                "a SpatialBatchLayout."
            )
        return layout

    @staticmethod
    def _validate_layout(
        layout: SpatialBatchLayout,
        *,
        activation_batch: int,
        activation_height: int,
        activation_width: int,
    ) -> None:
        """Require layout order and view geometry to match the active tensor."""

        if activation_batch % layout.expanded_batch_size != 0:
            raise ValueError(
                "Anima activation batch must be a whole number of conditioning "
                f"chunks over layout batch {layout.expanded_batch_size}; observed "
                f"batch {activation_batch}."
            )
        mismatched_views = tuple(
            index
            for index, view in enumerate(layout.views)
            if (view.model_height, view.model_width)
            != (activation_height, activation_width)
        )
        if mismatched_views:
            raise ValueError(
                "Anima activation H/W must match every spatial view model size; "
                f"mismatched view indices {mismatched_views}."
            )

    @staticmethod
    def _publish(
        transformer_options: dict[object, object],
        geometry: AnimaActivationGeometry,
    ) -> dict[object, object]:
        """Copy options and namespace before adding one activation value."""

        published = transformer_options.copy()
        existing_namespace = transformer_options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
        if existing_namespace is None:
            namespace: dict[object, object] = {}
        elif isinstance(existing_namespace, dict):
            namespace = existing_namespace.copy()
        else:
            raise TypeError("transformer_options.simple_syrup must be a dictionary.")
        if ANIMA_ACTIVATION_GEOMETRY_KEY in namespace:
            raise ValueError(
                "Anima activation geometry is already present in the SimpleSyrup "
                "transformer namespace."
            )
        namespace[ANIMA_ACTIVATION_GEOMETRY_KEY] = geometry
        published[SIMPLE_SYRUP_TRANSFORMER_NAMESPACE] = namespace
        return published


def anima_activation_wrapper_mutation(
    surface: AnimaModuleSurface,
    context: AnimaActivationContext,
) -> ModelDiffusionWrapperMutation:
    """Build the clone-local keyed mutation for one discovered Anima model."""

    return ModelDiffusionWrapperMutation(
        ANIMA_ACTIVATION_WRAPPER_KEY,
        AnimaActivationDiffusionWrapper(surface, context),
    )


def _ceil_div(value: int, divisor: int) -> int:
    """Return positive integer ceiling division for a padded patch grid."""

    return (value + divisor - 1) // divisor


ANIMA_ACTIVATION_CONTEXT = AnimaActivationContext()
