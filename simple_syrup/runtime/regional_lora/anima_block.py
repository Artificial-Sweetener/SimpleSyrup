# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assemble exact regional-LoRA block mutations for installed Anima graphs."""

from __future__ import annotations

from ..model_patcher_mutations import ModelExactObjectPatchMutation
from .anima_activation_context import AnimaActivationContext
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_block_execution import AnimaRegionalLoraBlockPatch
from .anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocationContext,
)
from .anima_module_surface import AnimaModuleSurface
from .anima_query_activity import (
    ANIMA_REGIONAL_QUERY_ACTIVITY_CONTEXT,
    AnimaRegionalQueryActivityContext,
)
from .anima_schedule_context import AnimaRegionalLoraScheduleContext


def anima_lora_block_mutations(
    surface: AnimaModuleSurface,
    attention: AnimaRegionalAttentionExecution,
    *,
    activation_context: AnimaActivationContext,
    branch_context: AnimaLoraBranchInvocationContext,
    spatial_context: AnimaLoraSpatialInvocationContext,
    schedule_context: AnimaRegionalLoraScheduleContext,
    query_activity: AnimaRegionalQueryActivityContext = (
        ANIMA_REGIONAL_QUERY_ACTIVITY_CONTEXT
    ),
) -> tuple[ModelExactObjectPatchMutation, ...]:
    """Build one exact regional block replacement per discovered Anima block."""

    return tuple(
        ModelExactObjectPatchMutation(
            path=f"diffusion_model.blocks.{block.block_index}",
            expected_object=block.block,
            replacement=AnimaRegionalLoraBlockPatch(
                block.block,
                attention,
                activation_context=activation_context,
                branch_context=branch_context,
                spatial_context=spatial_context,
                schedule_context=schedule_context,
                query_activity=query_activity,
            ),
        )
        for block in surface.blocks
    )
