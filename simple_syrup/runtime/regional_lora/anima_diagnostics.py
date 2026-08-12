# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose common Anima attention and regional LoRA diagnostics."""

from __future__ import annotations

from ...masking.regional_mask_activation import RegionalMaskActivationClassifier
from ..regional_attention_model_call_values import (
    regional_attention_model_call_values,
)
from ..regional_lora_schedule_resolution import RegionalLoraScheduleResolution
from .anima_activation_context import AnimaActivationGeometry
from .anima_attention_diagnostics import AnimaAttentionDiagnosticsBuilder
from .anima_composition import AnimaRegionalLoraComposition
from .anima_diagnostic_values import (
    AnimaRegionalExecutionDiagnostics,
    AnimaWorkEstimateDiagnostics,
)
from .anima_diagnostics_cache import AnimaDiagnosticsSnapshotCache
from .anima_lora_diagnostics import AnimaLoraDiagnosticsBuilder
from .anima_module_surface import AnimaModuleSurface
from .anima_query_masks import ANIMA_QUERY_MASK_PROJECTOR, AnimaQueryMaskProjector


class AnimaRegionalDiagnosticsBuilder:
    """Adapt Anima geometry and compose shared plus LoRA diagnostics."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        composition: AnimaRegionalLoraComposition,
        *,
        query_masks: AnimaQueryMaskProjector = ANIMA_QUERY_MASK_PROJECTOR,
        activation_classifier: RegionalMaskActivationClassifier | None = None,
    ) -> None:
        """Bind focused common, LoRA, layout, and cache collaborators."""

        if not isinstance(surface, AnimaModuleSurface):
            raise TypeError("Anima diagnostics require a verified module surface.")
        if not isinstance(composition, AnimaRegionalLoraComposition):
            raise TypeError("Anima diagnostics require a LoRA composition.")
        if not isinstance(query_masks, AnimaQueryMaskProjector):
            raise TypeError("Anima diagnostics require a query-mask projector.")
        self._composition = composition
        self._attention = AnimaAttentionDiagnosticsBuilder(
            surface,
            composition.attention,
            query_masks=query_masks,
            activation_classifier=activation_classifier,
        )
        self._lora = AnimaLoraDiagnosticsBuilder(surface, composition)
        self._cache = AnimaDiagnosticsSnapshotCache()

    def build(
        self,
        geometry: AnimaActivationGeometry,
        *,
        transformer_options: object | None = None,
        schedule_resolution: RegionalLoraScheduleResolution,
    ) -> AnimaRegionalExecutionDiagnostics:
        """Build one call snapshot using the exact resolved spatial layout."""

        layout = self._attention.resolve_layout(geometry)
        contexts = self._composition.attention.active_contexts
        call_values = regional_attention_model_call_values(transformer_options)
        self._attention.validate_call(geometry, transformer_options=transformer_options)
        lora = self._lora.build(schedule_resolution)
        cache_key = (
            geometry,
            layout,
            schedule_resolution,
            lora.prepared_cache_entries,
            id(contexts),
            call_values,
        )
        existing = self._cache.get(cache_key)
        if existing is not None:
            return existing
        regional = self._attention.build(
            geometry,
            transformer_options=transformer_options,
        )
        return self._cache.store(
            cache_key,
            AnimaRegionalExecutionDiagnostics.from_regional(
                regional,
                adapter_uses=lora.adapter_uses,
                prepared_cache_entries=lora.prepared_cache_entries,
                work=AnimaWorkEstimateDiagnostics(
                    cross_attention_branch_multiplier=(
                        regional.cross_attention_branch_multiplier
                    ),
                    low_rank_adapter_multiplier=(lora.work.low_rank_adapter_multiplier),
                    active_adapter_uses=lora.work.active_adapter_uses,
                    active_target_count=lora.work.active_target_count,
                    target_use_count=lora.work.target_use_count,
                    deduplicated_target_group_count=(
                        lora.work.deduplicated_target_group_count
                    ),
                    compatible_projection_batch_count=(
                        lora.work.compatible_projection_batch_count
                    ),
                    deduplicated_target_uses=lora.work.deduplicated_target_uses,
                    denoiser_call_multiplier=regional.denoiser_call_multiplier,
                ),
            ),
        )

    def validate_call(
        self,
        geometry: AnimaActivationGeometry,
        *,
        transformer_options: object | None = None,
    ) -> None:
        """Validate dynamic geometry and metadata without building a log payload."""

        self._attention.validate_call(
            geometry,
            transformer_options=transformer_options,
        )
