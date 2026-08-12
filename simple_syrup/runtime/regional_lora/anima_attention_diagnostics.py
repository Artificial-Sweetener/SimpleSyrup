# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt Anima execution state to shared regional diagnostics."""

from __future__ import annotations

from ...domain.spatial_views import SpatialBatchLayout
from ...masking.regional_mask_activation import RegionalMaskActivationClassifier
from ..regional_attention_diagnostic_values import (
    RegionalAttentionExecutionDiagnostics,
    RegionalAttentionQueryGeometry,
)
from ..regional_attention_diagnostics import RegionalAttentionDiagnosticsBuilder
from .anima_activation_context import AnimaActivationGeometry
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_module_surface import AnimaModuleSurface
from .anima_query_masks import (
    ANIMA_QUERY_MASK_PROJECTOR,
    AnimaQueryMaskProjector,
)


class AnimaAttentionDiagnosticsBuilder:
    """Build common regional evidence from authoritative Anima call state."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        attention: AnimaRegionalAttentionExecution,
        *,
        query_masks: AnimaQueryMaskProjector = ANIMA_QUERY_MASK_PROJECTOR,
        activation_classifier: RegionalMaskActivationClassifier | None = None,
    ) -> None:
        """Bind the verified Anima, attention, layout, and common authorities."""

        if not isinstance(surface, AnimaModuleSurface):
            raise TypeError("Anima diagnostics require a verified module surface.")
        if not isinstance(attention, AnimaRegionalAttentionExecution):
            raise TypeError("Anima diagnostics require an attention execution.")
        if not isinstance(query_masks, AnimaQueryMaskProjector):
            raise TypeError("Anima diagnostics require a query-mask projector.")
        model_type = type(surface.diffusion_model)
        self._attention = attention
        self._query_masks = query_masks
        self._regional = RegionalAttentionDiagnosticsBuilder(
            attention.mask_bank,
            backend=f"{model_type.__module__}.{model_type.__qualname__}",
            activation_classifier=activation_classifier,
        )

    def build(
        self,
        geometry: AnimaActivationGeometry,
        *,
        transformer_options: object | None = None,
    ) -> RegionalAttentionExecutionDiagnostics:
        """Build one common snapshot using the exact resolved spatial layout."""

        return self._regional.build(
            self._attention.active_contexts,
            self._query_geometry(geometry),
            self.resolve_layout(geometry),
            transformer_options=transformer_options,
        )

    def validate_call(
        self,
        geometry: AnimaActivationGeometry,
        *,
        transformer_options: object | None = None,
    ) -> None:
        """Validate dynamic geometry and metadata without building a payload."""

        self._regional.validate_call(
            self._attention.active_contexts,
            self._query_geometry(geometry),
            self.resolve_layout(geometry),
            transformer_options=transformer_options,
        )

    def resolve_layout(self, geometry: AnimaActivationGeometry) -> SpatialBatchLayout:
        """Resolve Anima's authoritative spatial layout for one call."""

        if not isinstance(geometry, AnimaActivationGeometry):
            raise TypeError("Anima diagnostics require activation geometry.")
        return self._query_masks.resolve_layout(
            self._attention.mask_bank,
            geometry,
            latent_batch_size=self._attention.active_contexts.latent_batch_size,
        )

    @staticmethod
    def _query_geometry(
        geometry: AnimaActivationGeometry,
    ) -> RegionalAttentionQueryGeometry:
        """Adapt Anima geometry to the shared diagnostic value."""

        return RegionalAttentionQueryGeometry(
            input_batch_size=geometry.input_batch_size,
            query_time=geometry.query_time,
            query_height=geometry.query_height,
            query_width=geometry.query_width,
            spatial_layout=geometry.spatial_layout,
        )
