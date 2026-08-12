# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive the complete full-context Anima Attention Coupling model."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..patcher_lifecycle import PATCHER_LIFECYCLE
from ..regional_attention_template import build_regional_attention_template
from ..regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from .anima_attention_context_wrapper import (
    anima_attention_context_wrapper_mutation,
)
from .anima_attention_coupling import anima_attention_coupling_mutations
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_composition import AnimaRegionalLoraComposition
from .anima_execution_scope import AnimaRegionalLoraAdapterExecution
from .anima_global_lora_overlap import ANIMA_GLOBAL_REGIONAL_LORA_OVERLAP_VALIDATOR
from .anima_model_patcher_surface import ANIMA_MODEL_PATCHER_SURFACE_RESOLVER
from .anima_plan_admission import ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE
from .execution_cache import ModelCloneLineage, RegionalLoraExecutionCache


@dataclass(frozen=True, slots=True)
class FullContextAnimaAttentionModel:
    """Return one derived model with its immutable attention execution owner."""

    model: object
    attention: AnimaRegionalAttentionExecution


class FullContextAnimaAttentionBackend:
    """Compose admitted regional attention and LoRA mutations for Anima."""

    def derive(
        self,
        *,
        model: object,
        processed_plan: ProcessedRegionalAttentionPlan,
        adaptation: RegionalLoraPlanAdaptation,
        region_strengths: tuple[float, ...],
        latent_batch_size: int,
    ) -> FullContextAnimaAttentionModel:
        """Return one collision-safe clone prepared for dynamic sampler calls."""

        if not isinstance(processed_plan, ProcessedRegionalAttentionPlan):
            raise TypeError("Anima backend requires a processed attention plan.")
        if not isinstance(adaptation, RegionalLoraPlanAdaptation):
            raise TypeError("Anima backend requires a regional LoRA adaptation.")
        admitted = ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE.admit(adaptation)
        ANIMA_GLOBAL_REGIONAL_LORA_OVERLAP_VALIDATOR.validate(model, admitted)
        template = build_regional_attention_template(
            processed_plan,
            latent_batch_size=latent_batch_size,
        )
        attention = AnimaRegionalAttentionExecution(
            template,
            processed_plan.mask_bank,
            region_strengths,
            dynamic_contexts=True,
        )
        cache = RegionalLoraExecutionCache()
        model_lineage = ModelCloneLineage.from_model(model)
        executions = tuple(
            AnimaRegionalLoraAdapterExecution(
                adapter.adapter_plan,
                adapter.admission,
                attention,
                model_lineage,
                cache,
            )
            for adapter in admitted.adapters
        )
        composition = (
            None if not executions else AnimaRegionalLoraComposition(executions)
        )
        surface = ANIMA_MODEL_PATCHER_SURFACE_RESOLVER.resolve(model)
        mutations = (
            anima_attention_context_wrapper_mutation(
                surface,
                processed_plan,
                attention,
            ),
            *anima_attention_coupling_mutations(
                surface,
                attention,
                composition=composition,
            ),
        )
        derived = PATCHER_LIFECYCLE.derive_model(
            model,
            mutations,
            operation="full-context Anima Attention Coupling",
        )
        return FullContextAnimaAttentionModel(derived, attention)


FULL_CONTEXT_ANIMA_ATTENTION_BACKEND = FullContextAnimaAttentionBackend()
