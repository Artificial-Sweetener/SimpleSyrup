# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify regional LoRA ownership inside compact standard-UNet attn2 branches."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import torch
import torch.nn.functional as functional
from torch import nn

from simple_syrup.domain.regional_activation_geometry import RegionalActivationLayout
from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraBranch,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.linear_execution import RegionalLinearExecutor
from simple_syrup.runtime.regional_lora.linear_execution_plan import (
    RegionalLinearExecutionPlan,
    RegionalLinearOperationKey,
    RegionalLinearTargetUse,
)
from simple_syrup.runtime.regional_lora.preparation import (
    RegionalLoraTargetPreparation,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget
from simple_syrup.runtime.regional_lora.standard_unet_packed_operation_masks import (
    StandardUnetPackedOperationMaskResolver,
)


@dataclass(frozen=True, slots=True)
class _Use:
    """Expose the operation ownership protocol for one test use."""

    composition_index: int
    region_index: int
    branch: RegionalLoraBranch


def test_packed_image_masks_exclude_base_and_preserve_cfg_branch_ownership() -> None:
    """Apply spatial masks only to the matching regional query/output rows."""

    execution = _execution()
    inputs = torch.zeros((4, 2, 8))

    masks = StandardUnetPackedOperationMaskResolver().resolve_image_tokens(
        execution,
        uses=(
            _Use(0, 0, RegionalLoraBranch.POSITIVE),
            _Use(1, 0, RegionalLoraBranch.NEGATIVE),
        ),
        inputs=inputs,
    )

    assert masks.geometry.layout is RegionalActivationLayout.CONSUMER_SPATIALIZED
    assert masks.multipliers.shape == (2, 4, 2, 1)
    torch.testing.assert_close(
        masks.multipliers[0, :, :, 0],
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.5], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        masks.multipliers[1, :, :, 0],
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.25, 1.0]]),
    )


def test_packed_context_masks_gate_rows_without_reshaping_clip_tokens() -> None:
    """Broadcast branch ownership over the native full context-token sequence."""

    execution = _execution()
    inputs = torch.zeros((4, 77, 64))

    masks = StandardUnetPackedOperationMaskResolver().resolve_context_tokens(
        execution,
        uses=(
            _Use(0, 0, RegionalLoraBranch.POSITIVE),
            _Use(1, 0, RegionalLoraBranch.NEGATIVE),
        ),
        inputs=inputs,
    )

    assert masks.geometry.layout is RegionalActivationLayout.BRANCH_TOKENS
    assert masks.geometry.invocation_shape == (4, 77, 64)
    assert masks.multipliers.shape == (2, 4, 77, 1)
    assert masks.multipliers[0, :2].eq(0.0).all()
    assert masks.multipliers[0, 2].eq(1.0).all()
    assert masks.multipliers[0, 3].eq(0.0).all()
    assert masks.multipliers[1, 3].eq(1.0).all()


def test_packed_image_and_native_context_operations_match_rank_reference() -> None:
    """Execute exact Q/output and K/V LoRA math on their native packed shapes."""

    execution = _execution()
    plan, target = _linear_plan()
    base = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[0.2, -0.1], [0.4, 0.3]]))
        base.bias.copy_(torch.tensor([0.05, -0.2]))
    schedules = (0.5, 0.25)

    for inputs, masks in (
        (
            torch.linspace(-1.0, 1.0, 16).reshape(4, 2, 2),
            StandardUnetPackedOperationMaskResolver().resolve_image_tokens(
                execution,
                uses=plan.uses,
                inputs=torch.zeros((4, 2, 2)),
            ),
        ),
        (
            torch.linspace(-1.0, 1.0, 4 * 77 * 2).reshape(4, 77, 2),
            StandardUnetPackedOperationMaskResolver().resolve_context_tokens(
                execution,
                uses=plan.uses,
                inputs=torch.zeros((4, 77, 2)),
            ),
        ),
    ):
        result = RegionalLinearExecutor().execute(
            base,
            inputs,
            plan=plan,
            masks=masks,
            schedule_strengths=schedules,
        )
        rank = functional.linear(inputs, target.down)
        combined = (
            masks.multipliers[0, :, :, 0] * schedules[0]
            + masks.multipliers[1, :, :, 0] * schedules[1]
        ) * target.intrinsic_scale
        expected = base(inputs) + functional.linear(
            rank * combined.unsqueeze(-1),
            target.up,
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def _execution() -> UnetAttn2Execution:
    """Return two CFG rows and one compact regional branch over both rows."""

    base = torch.zeros((2, 77, 64))
    contexts = BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=(
            RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.NEGATIVE, 1, 2),
        ),
        base_context=base,
        regions=(
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, base.clone(), (1.0, 1.0)),),
            ),
        ),
    )
    return UnetAttn2Execution(
        contexts,
        torch.tensor([[[1.0, 0.5], [0.25, 1.0]]]),
        (1.0,),
        1,
        2,
    )


def _linear_plan() -> tuple[RegionalLinearExecutionPlan, StandardLoraTarget]:
    """Return one exact adapter used by positive and negative packed branches."""

    identity = RegionalLoraAdapterIdentity("packed.safetensors")
    target = StandardLoraTarget(
        "diffusion_model.attn2.weight",
        torch.tensor([[0.3, -0.2], [0.1, 0.4]]),
        torch.tensor([[0.5, -0.1], [-0.3, 0.2]]),
        2,
        2,
        2,
        0.5,
    )
    cache = RegionalLoraExecutionCache()
    preparation = RegionalLoraTargetPreparation(
        identity,
        ModelCloneLineage(UUID(int=1), UUID(int=2)),
        target,
        cache,
    )
    key = RegionalLinearOperationKey(
        identity,
        target.target,
        id(target.down),
        id(target.up),
    )
    return (
        RegionalLinearExecutionPlan(
            (
                RegionalLinearTargetUse(
                    0,
                    0,
                    RegionalLoraBranch.POSITIVE,
                    key,
                    preparation,
                    target.intrinsic_scale,
                ),
                RegionalLinearTargetUse(
                    1,
                    0,
                    RegionalLoraBranch.NEGATIVE,
                    key,
                    preparation,
                    target.intrinsic_scale,
                ),
            )
        ),
        target,
    )
