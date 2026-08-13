# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build explicit static regional LoRA schedule state for focused tests."""

from __future__ import annotations

from uuid import uuid4

import torch
from regional_attention_test_values import single_entry_regions

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import AnimaQueryMaskBatch
from simple_syrup.runtime.regional_lora.anima_schedule_context import (
    AnimaRegionalLoraScheduleContext,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    AnimaLoraAdmission,
    AnimaLoraTarget,
    AnimaLoraTargetFamily,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)


def static_lora_schedule(
    composition: AnimaRegionalLoraComposition,
) -> tuple[AnimaRegionalLoraScheduleContext, RegionalLoraScheduleResolution]:
    """Return an explicit all-schedule-active state using each base strength."""

    strengths = tuple(
        execution.adapter_plan.model_strength for execution in composition.executions
    )
    return (
        AnimaRegionalLoraScheduleContext(),
        RegionalLoraScheduleResolution(
            schedule_multipliers=(1.0,) * len(strengths),
            effective_strengths=strengths,
        ),
    )


def single_target_execution(
    family: AnimaLoraTargetFamily,
    branch: RegionalLoraBranch,
) -> AnimaRegionalLoraAdapterExecution:
    """Build one deterministic two-feature execution across both CFG chunks."""

    target = _identity_target(family)
    plan = RegionalLoraAdapterPlan(
        adapter_identity=RegionalLoraAdapterIdentity("adapter.safetensors"),
        composition_index=0,
        region_index=0,
        branch=branch,
        model_strength=0.5,
        schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
    )
    base = torch.zeros((2, 1, 2))
    contexts = BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=(
            RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.NEGATIVE, 1, 2),
        ),
        base_context=base,
        regions=single_entry_regions((base.clone(),)),
    )
    mask = torch.tensor([[[0.25, 0.75]]])
    attention = AnimaRegionalAttentionExecution(
        contexts,
        RegionalMaskBank(mask.clone(), mask.clone(), 2, 1),
        (1.0,),
    )
    return AnimaRegionalLoraAdapterExecution(
        plan,
        AnimaLoraAdmission((target,)),
        attention,
        ModelCloneLineage(uuid4(), uuid4()),
        RegionalLoraExecutionCache(),
    )


def single_region_query_masks() -> AnimaQueryMaskBatch:
    """Return one feathered region repeated across both CFG chunks."""

    return AnimaQueryMaskBatch(torch.tensor([[[[[0.25, 0.75]]], [[[0.25, 0.75]]]]]))


def _identity_target(family: AnimaLoraTargetFamily) -> AnimaLoraTarget:
    """Build one rank-two identity adapter for a synthetic two-feature target."""

    adapter = StandardLoraTarget(
        f"diffusion_model.blocks.0.{family.value}",
        torch.eye(2),
        torch.eye(2),
        rank=2,
        input_features=2,
        output_features=2,
    )
    return AnimaLoraTarget(0, family, adapter)
