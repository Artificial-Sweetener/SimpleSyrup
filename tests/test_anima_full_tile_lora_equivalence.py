# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove full-context and full-canvas one-tile Anima LoRA equivalence."""

from __future__ import annotations

from uuid import uuid4

import pytest
import torch
from regional_attention_test_values import single_entry_regions
from torch import nn

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
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocation,
    AnimaLoraSpatialInvocationContext,
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_linear_execution import (
    AnimaRegionalLoraCompositionLinearPatch,
)
from simple_syrup.runtime.regional_lora.anima_lora_weights import (
    AnimaLoraWeightResolver,
)
from simple_syrup.runtime.regional_lora.anima_query_activity import (
    AnimaRegionalQueryActivityContext,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import (
    AnimaQueryMaskBatch,
)
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
    RegionalLoraScheduleSession,
)


class _ZeroLinear(nn.Module):
    """Return an exact zero original output for isolated LoRA comparison."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Preserve the leading shape and scalar output width."""

        return torch.zeros_like(inputs)


def test_full_and_one_tile_share_exact_attention_and_adaln_activity() -> None:
    """Preserve masks, weights, and compact branches across both geometries."""

    execution = _attention()
    activity_context = AnimaRegionalQueryActivityContext()

    full = activity_context.resolve(
        execution,
        _full_geometry(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    tiled = activity_context.resolve(
        execution,
        _one_tile_geometry(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(full.masks.masks, tiled.masks.masks)
    torch.testing.assert_close(
        full.attention_weights.base, tiled.attention_weights.base
    )
    torch.testing.assert_close(
        full.attention_weights.regions,
        tiled.attention_weights.regions,
    )
    assert full.attention_branches.invocation == tiled.attention_branches.invocation
    assert full.adaln_branches.invocation == tiled.adaln_branches.invocation


@pytest.mark.parametrize("multi_lora", [False, True], ids=("single", "multi"))
def test_scheduled_lora_outputs_match_full_and_one_tile(
    multi_lora: bool,
) -> None:
    """Retain schedules, repeated regions, order, and CFG gates in one tile."""

    attention = _attention()
    executions = _executions(attention, multi_lora=multi_lora)
    composition = AnimaRegionalLoraComposition(executions)
    spatial = AnimaLoraSpatialInvocationContext()
    schedule = AnimaRegionalLoraScheduleContext()
    target_name = executions[0].admission.targets[0].adapter.target
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _ZeroLinear(),
        composition.groups_for_target(target_name),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=spatial,
        ),
        schedule_context=schedule,
    )
    activity = AnimaRegionalQueryActivityContext()
    full_masks = activity.resolve(
        attention,
        _full_geometry(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).masks
    tile_masks = activity.resolve(
        attention,
        _one_tile_geometry(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).masks
    inputs = torch.arange(1.0, 17.0).reshape(2, 8, 1)
    session = RegionalLoraScheduleSession(
        tuple(execution.adapter_plan for execution in executions),
        maximum_sigma=100.0,
    )

    for sigma in (100.0, 75.0, 50.0, 25.0, 0.0):
        resolution = session.resolve(sigma)
        full_output = _execute(
            patch,
            inputs,
            masks=full_masks,
            spatial=spatial,
            schedule=schedule,
            resolution=resolution,
        )
        tile_output = _execute(
            patch,
            inputs,
            masks=tile_masks,
            spatial=spatial,
            schedule=schedule,
            resolution=resolution,
        )
        torch.testing.assert_close(full_output, tile_output, rtol=0.0, atol=0.0)

    if multi_lora:
        assert (
            executions[0].adapter_plan.adapter_identity
            == executions[1].adapter_plan.adapter_identity
        )
        assert executions[0].adapter_plan.region_index == 0
        assert executions[1].adapter_plan.region_index == 1
        assert executions[2].adapter_plan.branch is RegionalLoraBranch.NEGATIVE


def test_contextual_global_uses_same_lora_plan_with_projected_canvas_masks() -> None:
    """Keep adapter ownership and schedules while reducing the full source view."""

    attention = _attention()
    executions = _executions(attention, multi_lora=True)
    composition = AnimaRegionalLoraComposition(executions)
    activity = AnimaRegionalQueryActivityContext().resolve(
        attention,
        _contextual_global_geometry(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    target_name = executions[0].admission.targets[0].adapter.target
    groups = composition.groups_for_target(target_name)
    session = RegionalLoraScheduleSession(
        tuple(execution.adapter_plan for execution in executions),
        maximum_sigma=100.0,
    )

    torch.testing.assert_close(
        activity.masks.masks,
        torch.full((2, 2, 1, 1, 1), 0.625),
    )
    assert tuple(use.execution for group in groups for use in group.uses) == executions
    assert tuple(
        execution.adapter_plan.adapter_identity for execution in executions
    ) == (
        RegionalLoraAdapterIdentity("shared.safetensors"),
        RegionalLoraAdapterIdentity("shared.safetensors"),
        RegionalLoraAdapterIdentity("negative.safetensors"),
    )
    assert tuple(
        execution.admission.targets[0].adapter.target for execution in executions
    ) == (target_name, target_name, target_name)
    assert tuple(
        session.resolve(sigma).effective_strengths
        for sigma in (100.0, 75.0, 50.0, 25.0, 0.0)
    ) == (
        (0.75, 0.0, -0.5),
        (0.75, 0.75, -0.5),
        (0.375, 0.75, 0.5),
        (0.375, 0.1875, 0.5),
        (0.0, 0.1875, 0.5),
    )


def _execute(
    patch: AnimaRegionalLoraCompositionLinearPatch,
    inputs: torch.Tensor,
    *,
    masks: AnimaQueryMaskBatch,
    spatial: AnimaLoraSpatialInvocationContext,
    schedule: AnimaRegionalLoraScheduleContext,
    resolution: object,
) -> torch.Tensor:
    """Execute one pointwise composition under paired runtime authorities."""

    from simple_syrup.runtime.regional_lora_schedule_resolution import (
        RegionalLoraScheduleResolution,
    )

    if not isinstance(resolution, RegionalLoraScheduleResolution):
        raise TypeError("Equivalence test requires a schedule resolution.")
    with (
        schedule.activate(resolution),
        spatial.activate(AnimaLoraSpatialInvocation(masks)),
    ):
        output = patch(inputs)
    if not isinstance(output, torch.Tensor):
        raise TypeError("Equivalence LoRA patch must return a tensor.")
    return output


def _executions(
    attention: AnimaRegionalAttentionExecution,
    *,
    multi_lora: bool,
) -> tuple[AnimaRegionalLoraAdapterExecution, ...]:
    """Build one adapter or repeated-region multi-LoRA CFG composition."""

    shared = _target(down=1.25, up=0.5)
    uses: tuple[
        tuple[
            AnimaLoraTarget,
            str,
            int,
            RegionalLoraBranch,
            float,
            tuple[RegionalLoraScheduleBoundary, ...],
        ],
        ...,
    ] = (
        (
            shared,
            "shared.safetensors",
            0,
            RegionalLoraBranch.POSITIVE,
            0.75,
            _schedule((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)),
        ),
    )
    if multi_lora:
        uses = (
            uses[0],
            (
                shared,
                "shared.safetensors",
                1,
                RegionalLoraBranch.POSITIVE,
                0.75,
                _schedule((0.0, 0.0), (0.25, 1.0), (0.75, 0.25)),
            ),
            (
                _target(down=-0.75, up=2.0),
                "negative.safetensors",
                0,
                RegionalLoraBranch.NEGATIVE,
                -0.5,
                _schedule((0.0, 1.0), (0.5, -1.0)),
            ),
        )
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    return tuple(
        AnimaRegionalLoraAdapterExecution(
            RegionalLoraAdapterPlan(
                RegionalLoraAdapterIdentity(identity),
                composition_index=index,
                region_index=region,
                branch=branch,
                model_strength=strength,
                schedule=boundaries,
            ),
            AnimaLoraAdmission((target,)),
            attention,
            model,
            cache,
        )
        for index, (
            target,
            identity,
            region,
            branch,
            strength,
            boundaries,
        ) in enumerate(uses)
    )


def _target(*, down: float, up: float) -> AnimaLoraTarget:
    """Build one scalar self-attention target with recognizable weights."""

    target_name = "diffusion_model.blocks.0.self_attn.q_proj"
    return AnimaLoraTarget(
        0,
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        StandardLoraTarget(
            target_name,
            torch.tensor([[down]]),
            torch.tensor([[up]]),
            rank=1,
            input_features=1,
            output_features=1,
        ),
    )


def _schedule(
    *values: tuple[float, float],
) -> tuple[RegionalLoraScheduleBoundary, ...]:
    """Build canonical boundaries under a linear 100-to-zero sigma schedule."""

    return tuple(
        RegionalLoraScheduleBoundary(
            percent,
            100.0 * (1.0 - percent),
            multiplier,
            0,
        )
        for percent, multiplier in values
    )


def _attention() -> AnimaRegionalAttentionExecution:
    """Build two overlapping regions over ordinary positive/negative CFG."""

    context = torch.zeros((2, 1, 1))
    contexts = BatchedRegionalAttentionContexts(
        1,
        (
            RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.NEGATIVE, 1, 2),
        ),
        context,
        single_entry_regions((context.clone(), context.clone())),
    )
    masks = torch.tensor(
        [
            [[1.0, 1.0, 0.5, 0.0], [1.0, 1.0, 0.5, 0.0]],
            [[0.0, 0.5, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0]],
        ]
    )
    return AnimaRegionalAttentionExecution(
        contexts,
        RegionalMaskBank(masks.clone(), masks, 4, 2),
        (1.0, 1.0),
    )


def _full_geometry() -> AnimaActivationGeometry:
    """Return one full-context two-chunk query geometry."""

    return AnimaActivationGeometry(2, 1, 2, 4, 1, 1, 1, 2, 4, None)


def _one_tile_geometry() -> AnimaActivationGeometry:
    """Return the same canvas as one explicit full-canvas tile view."""

    layout = SpatialBatchLayout(
        4,
        2,
        (SpatialView(SpatialViewKind.TILE, 0, 0, 4, 2, 4, 2),),
        input_batch_size=2,
    )
    return AnimaActivationGeometry(2, 1, 2, 4, 1, 1, 1, 2, 4, layout)


def _contextual_global_geometry() -> AnimaActivationGeometry:
    """Return one full-source view reduced to a single Anima query token."""

    layout = SpatialBatchLayout(
        4,
        2,
        (
            SpatialView(
                SpatialViewKind.CONTEXTUAL_GLOBAL,
                0,
                0,
                4,
                2,
                2,
                2,
            ),
        ),
        input_batch_size=2,
    )
    return AnimaActivationGeometry(2, 1, 2, 2, 1, 2, 1, 1, 1, layout)
