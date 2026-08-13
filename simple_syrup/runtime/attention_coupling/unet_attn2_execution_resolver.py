# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define standard-UNet per-layer attn2 execution resolution boundaries."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
)
from ..regional_attention_query_masks import (
    REGIONAL_ATTENTION_QUERY_MASK_PROJECTOR,
    RegionalAttentionQueryMaskProjector,
)
from .unet_attention_diagnostics import (
    STANDARD_UNET_ATTENTION_DIAGNOSTICS_EMITTER,
    StandardUnetAttentionDiagnosticsEmitter,
)
from .unet_attention_state import StandardUnetAttentionState
from .unet_attn2_execution import UnetAttn2Execution
from .unet_attn2_geometry import (
    STANDARD_UNET_ATTN2_GEOMETRY_RESOLVER,
    StandardUnetAttn2Geometry,
    StandardUnetAttn2GeometryResolver,
)
from .unet_attn2_resolution_cache import StandardUnetAttn2ResolutionKey


@runtime_checkable
class UnetAttn2ExecutionResolver(Protocol):
    """Resolve one native attention layer to its exact numerical execution."""

    def resolve(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> UnetAttn2Execution:
        """Return the execution matching the active call and query geometry."""


class StaticUnetAttn2ExecutionResolver:
    """Return one fixed execution for deterministic low-level backend harnesses."""

    def __init__(self, execution: UnetAttn2Execution) -> None:
        """Retain the exact preprojected P8.1 execution."""

        if not isinstance(execution, UnetAttn2Execution):
            raise TypeError("Static UNet resolver requires an attn2 execution.")
        self._execution = execution

    def resolve(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> UnetAttn2Execution:
        """Return the fixed execution after preserving callback arguments."""

        del query, context, extra_options
        return self._execution


class StandardUnetAttn2ExecutionResolver:
    """Resolve dynamic contexts and projected masks for one UNet resolution."""

    def __init__(
        self,
        state: StandardUnetAttentionState,
        *,
        geometry_resolver: StandardUnetAttn2GeometryResolver = (
            STANDARD_UNET_ATTN2_GEOMETRY_RESOLVER
        ),
        query_masks: RegionalAttentionQueryMaskProjector = (
            REGIONAL_ATTENTION_QUERY_MASK_PROJECTOR
        ),
        diagnostics: StandardUnetAttentionDiagnosticsEmitter = (
            STANDARD_UNET_ATTENTION_DIAGNOSTICS_EMITTER
        ),
    ) -> None:
        """Retain exact state, geometry, projection, and emission authorities."""

        if not isinstance(state, StandardUnetAttentionState):
            raise TypeError("Standard UNet resolver requires attention state.")
        if not isinstance(geometry_resolver, StandardUnetAttn2GeometryResolver):
            raise TypeError("Standard UNet resolver requires a geometry resolver.")
        if not isinstance(query_masks, RegionalAttentionQueryMaskProjector):
            raise TypeError("Standard UNet resolver requires a mask projector.")
        if not isinstance(diagnostics, StandardUnetAttentionDiagnosticsEmitter):
            raise TypeError("Standard UNet resolver requires diagnostics emission.")
        self._state = state
        self._geometry_resolver = geometry_resolver
        self._query_masks = query_masks
        self._diagnostics = diagnostics

    @property
    def state(self) -> StandardUnetAttentionState:
        """Return the exact state required by the backend."""

        return self._state

    def resolve(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> UnetAttn2Execution:
        """Return one cached exact execution for the active call and resolution."""

        contexts = self._state.execution_context.require_current()
        if context is not contexts.base_context:
            raise ValueError(
                "Standard UNet attn2 context must be the active base authority."
            )
        if query.device != context.device or query.dtype != context.dtype:
            raise ValueError(
                "Standard UNet attn2 query and context must share device and dtype."
            )
        geometry = self._geometry_resolver.resolve(
            query,
            contexts,
            self._state.plan.mask_bank,
            extra_options,
        )
        key = StandardUnetAttn2ResolutionKey.from_geometry(geometry, query)
        return self._state.resolution_cache.resolve(
            key,
            lambda: self._build(query, contexts, geometry, extra_options),
        )

    def _build(
        self,
        query: torch.Tensor,
        contexts: BatchedRegionalAttentionContexts,
        geometry: StandardUnetAttn2Geometry,
        extra_options: dict[str, Any],
    ) -> UnetAttn2Execution:
        """Project and diagnose one unique call-local resolution."""

        projected = self._query_masks.project(
            bank=self._state.plan.mask_bank,
            query_geometry=geometry.query,
            layout=geometry.layout,
            form=RegionalMaskForm.CONDITIONING,
            mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
            device=query.device,
            dtype=query.dtype,
        )
        execution = UnetAttn2Execution(
            contexts,
            projected.flattened,
            self._state.region_strengths,
            geometry.query.query_height,
            geometry.query.query_width,
        )
        snapshot = self._state.diagnostics.build(
            contexts,
            geometry.query,
            geometry.layout,
            transformer_options=extra_options,
        )
        self._diagnostics.emit(snapshot, geometry)
        return execution
