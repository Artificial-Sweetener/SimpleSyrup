# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Align and publish scheduled regional contexts around each Anima call."""

from __future__ import annotations

import torch

from ...domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..diffusion_wrapper_executor import DiffusionWrapperExecutor
from ..model_patcher_mutations import ModelDiffusionWrapperMutation
from ..regional_attention_model_call import (
    REGIONAL_ATTENTION_MODEL_CALL_RESOLVER,
    RegionalAttentionModelCallResolver,
)
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_model_reference import AnimaDiffusionModelReference
from .anima_module_surface import AnimaModuleSurface

ANIMA_ATTENTION_CONTEXT_WRAPPER_KEY = "simple_syrup.anima_regional_attention_contexts"


class AnimaRegionalAttentionContextDiffusionWrapper:
    """Publish the exact active CFG and schedule context batch per model call."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        plan: ProcessedRegionalAttentionPlan,
        execution: AnimaRegionalAttentionExecution,
        *,
        model_call_resolver: RegionalAttentionModelCallResolver = (
            REGIONAL_ATTENTION_MODEL_CALL_RESOLVER
        ),
    ) -> None:
        """Retain one processed plan and its task-local execution authority."""

        if not isinstance(surface, AnimaModuleSurface):
            raise TypeError("Anima context wrapper requires a module surface.")
        if not isinstance(plan, ProcessedRegionalAttentionPlan):
            raise TypeError("Anima context wrapper requires a processed plan.")
        if not isinstance(execution, AnimaRegionalAttentionExecution):
            raise TypeError("Anima context wrapper requires an attention execution.")
        if not execution.dynamic_contexts:
            raise ValueError("Anima context wrapper requires dynamic execution state.")
        self._model = AnimaDiffusionModelReference(surface.diffusion_model)
        self._plan = plan
        self._execution = execution
        if not isinstance(model_call_resolver, RegionalAttentionModelCallResolver):
            raise TypeError("Anima context wrapper requires a model-call resolver.")
        self._model_call_resolver = model_call_resolver

    def __call__(
        self,
        executor: DiffusionWrapperExecutor,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Align current base chunks and execute under their regional contexts."""

        if not self._model.owns(executor.class_obj):
            raise ValueError(
                "Anima context wrapper executor does not own the discovered model."
            )
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("Anima context wrapper requires a tensor model input.")
        model_input = args[0]
        if len(args) < 3 or not isinstance(args[2], torch.Tensor):
            raise TypeError(
                "Anima context wrapper requires a tensor third positional base context."
            )
        base_context = args[2]
        contexts = self._model_call_resolver.resolve(
            self._plan,
            model_input=model_input,
            base_context=base_context,
            transformer_options=kwargs.get("transformer_options"),
        )
        with self._execution.execution_context.activate(contexts):
            return executor(*args, **kwargs)


def anima_attention_context_wrapper_mutation(
    surface: AnimaModuleSurface,
    plan: ProcessedRegionalAttentionPlan,
    execution: AnimaRegionalAttentionExecution,
) -> ModelDiffusionWrapperMutation:
    """Return the collision-safe dynamic context wrapper mutation."""

    return ModelDiffusionWrapperMutation(
        ANIMA_ATTENTION_CONTEXT_WRAPPER_KEY,
        AnimaRegionalAttentionContextDiffusionWrapper(surface, plan, execution),
    )
