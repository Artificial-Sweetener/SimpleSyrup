# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose clone-local Anima Attention Coupling and optional LoRA mutations."""

from __future__ import annotations

from ..patcher_lifecycle import ModelMutation
from .anima_activation_context import (
    ANIMA_ACTIVATION_CONTEXT,
    AnimaActivationContext,
    anima_activation_wrapper_mutation,
)
from .anima_attention_device_cache_lifecycle import (
    AnimaAttentionDeviceCacheLifecycle,
)
from .anima_attention_diagnostics import AnimaAttentionDiagnosticsBuilder
from .anima_attention_diagnostics_wrapper import (
    anima_attention_diagnostics_wrapper_mutation,
)
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_block import anima_lora_block_mutations
from .anima_composition import AnimaRegionalLoraComposition
from .anima_composition_phase_context import (
    ANIMA_COMPOSITION_PHASE_CONTEXT,
    AnimaCompositionPhaseContext,
)
from .anima_composition_phase_wrapper import (
    anima_composition_phase_wrapper_mutation,
)
from .anima_cross_attention import anima_cross_attention_mutations
from .anima_cross_attention_context import (
    ANIMA_CROSS_ATTENTION_INVOCATION_CONTEXT,
    AnimaCrossAttentionInvocationContext,
)
from .anima_diagnostics import AnimaRegionalDiagnosticsBuilder
from .anima_diagnostics_wrapper import anima_diagnostics_wrapper_mutation
from .anima_execution_scope import (
    ANIMA_LORA_BRANCH_CONTEXT,
    ANIMA_LORA_SPATIAL_CONTEXT,
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocationContext,
)
from .anima_linear_mutations import anima_lora_composition_linear_mutations
from .anima_lora_weights import AnimaLoraWeightResolver
from .anima_module_surface import AnimaModuleSurface
from .anima_query_activity import AnimaRegionalQueryActivityContext
from .anima_query_mask_context import (
    ANIMA_QUERY_MASK_CONTEXT,
    AnimaQueryMaskContext,
)
from .anima_schedule_context import (
    ANIMA_REGIONAL_LORA_SCHEDULE_CONTEXT,
    AnimaRegionalLoraScheduleContext,
)
from .anima_schedule_wrapper import anima_schedule_wrapper_mutations
from .anima_self_attention import anima_self_attention_mutations
from .anima_self_attention_partition_cache import AnimaSelfAttentionPartitionCache


def anima_attention_coupling_mutations(
    surface: AnimaModuleSurface,
    attention: AnimaRegionalAttentionExecution,
    *,
    composition: AnimaRegionalLoraComposition | None = None,
    activation_context: AnimaActivationContext = ANIMA_ACTIVATION_CONTEXT,
    cross_attention_context: AnimaCrossAttentionInvocationContext = (
        ANIMA_CROSS_ATTENTION_INVOCATION_CONTEXT
    ),
    branch_context: AnimaLoraBranchInvocationContext = ANIMA_LORA_BRANCH_CONTEXT,
    spatial_context: AnimaLoraSpatialInvocationContext = ANIMA_LORA_SPATIAL_CONTEXT,
    schedule_context: AnimaRegionalLoraScheduleContext = (
        ANIMA_REGIONAL_LORA_SCHEDULE_CONTEXT
    ),
    phase_context: AnimaCompositionPhaseContext = (ANIMA_COMPOSITION_PHASE_CONTEXT),
    query_mask_context: AnimaQueryMaskContext = ANIMA_QUERY_MASK_CONTEXT,
) -> tuple[ModelMutation, ...]:
    """Return attention-only or complete regional-LoRA mutation composition."""

    if not isinstance(surface, AnimaModuleSurface):
        raise TypeError("Anima Attention Coupling requires an Anima surface.")
    if not isinstance(attention, AnimaRegionalAttentionExecution):
        raise TypeError("Anima Attention Coupling requires an attention execution.")
    if composition is not None and not isinstance(
        composition,
        AnimaRegionalLoraComposition,
    ):
        raise TypeError("Anima Attention Coupling composition has an invalid type.")
    if composition is not None and composition.attention is not attention:
        raise ValueError(
            "Anima Attention Coupling composition must share its attention owner."
        )

    activation = anima_activation_wrapper_mutation(surface, activation_context)
    query_activity = AnimaRegionalQueryActivityContext(query_mask_context)
    partition_cache = AnimaSelfAttentionPartitionCache()
    attention_cache_lifecycle = AnimaAttentionDeviceCacheLifecycle(
        query_activity,
        query_mask_context,
    )
    cross_mutations = anima_cross_attention_mutations(
        surface,
        attention,
        activation_context=activation_context,
        invocation_context=cross_attention_context,
        phase_context=phase_context,
        query_activity=query_activity,
    )
    phase_wrapper = anima_composition_phase_wrapper_mutation(
        surface,
        phase_context,
    )
    self_mutations = anima_self_attention_mutations(
        surface,
        attention,
        activation_context=activation_context,
        phase_context=phase_context,
        query_activity=query_activity,
        partition_cache=partition_cache,
    )
    if composition is None:
        attention_diagnostics = AnimaAttentionDiagnosticsBuilder(surface, attention)
        return (
            phase_wrapper,
            activation,
            anima_attention_diagnostics_wrapper_mutation(
                surface,
                activation_context,
                attention_diagnostics,
                phase_context,
            ),
            *self_mutations,
            *cross_mutations,
            attention_cache_lifecycle.mutation(),
            partition_cache.mutation(),
        )

    block_mutations = anima_lora_block_mutations(
        surface,
        attention,
        activation_context=activation_context,
        branch_context=branch_context,
        spatial_context=spatial_context,
        schedule_context=schedule_context,
        query_activity=query_activity,
    )
    weights = AnimaLoraWeightResolver(
        cross_attention_context=cross_attention_context,
        branch_context=branch_context,
        spatial_context=spatial_context,
    )
    linear_mutations = anima_lora_composition_linear_mutations(
        surface,
        composition,
        weight_resolver=weights,
        schedule_context=schedule_context,
    )
    schedule_wrapper, schedule_cache_lifecycle = anima_schedule_wrapper_mutations(
        surface,
        composition,
        schedule_context,
        phase_context,
    )
    regional_diagnostics = AnimaRegionalDiagnosticsBuilder(surface, composition)
    return (
        phase_wrapper,
        schedule_wrapper,
        activation,
        anima_diagnostics_wrapper_mutation(
            surface,
            activation_context,
            regional_diagnostics,
            schedule_context,
            phase_context,
        ),
        *block_mutations,
        *self_mutations,
        *cross_mutations,
        *linear_mutations,
        attention_cache_lifecycle.mutation(),
        schedule_cache_lifecycle,
        partition_cache.mutation(),
    )
