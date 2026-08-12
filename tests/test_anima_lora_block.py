# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove the Anima block retains one trajectory with exact regional AdaLN."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
import torch
from regional_attention_test_values import single_entry_regions
from regional_lora_test_values import static_lora_schedule
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
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationContext,
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_block_execution import (
    AnimaRegionalLoraBlockPatch,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocationContext,
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_linear_execution import (
    AnimaRegionalLoraCompositionLinearPatch,
)
from simple_syrup.runtime.regional_lora.anima_lora_weights import (
    AnimaLoraWeightResolver,
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


class _CountingAttention(nn.Module):
    """Return one constant attention result and retain exact call count."""

    def __init__(self, value: float) -> None:
        """Store the result value and start with no calls."""

        super().__init__()
        self.value = value
        self.calls = 0

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return one B/Q/D tensor matching the attention input."""

        del context, kwargs
        self.calls += 1
        return torch.full_like(inputs, self.value)


class _CountingMlp(nn.Module):
    """Capture the patched input and return a zero residual update."""

    def __init__(self) -> None:
        """Start with no calls or captured input."""

        super().__init__()
        self.calls = 0
        self.inputs: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Capture one input while returning zeros."""

        self.calls += 1
        self.inputs = inputs.clone()
        return torch.zeros_like(inputs)


class _TinyBlock(nn.Module):
    """Expose the installed Anima block child contract at one feature."""

    def __init__(self) -> None:
        """Create deterministic normalization, modulation, and stage modules."""

        super().__init__()
        self.layer_norm_self_attn = nn.Identity()
        self.layer_norm_cross_attn = nn.Identity()
        self.layer_norm_mlp = nn.Identity()
        self.self_attn = _CountingAttention(1.0)
        self.cross_attn = _CountingAttention(0.0)
        self.mlp = _CountingMlp()
        self.adaln_modulation_self_attn = _modulation()
        self.adaln_modulation_cross_attn = _modulation()
        self.adaln_modulation_mlp = _modulation()


def test_block_backing_module_does_not_leak_a_host_weight_namespace() -> None:
    """Keep the retained installed block outside PyTorch child discovery."""

    original = _TinyBlock()
    execution = _execution()
    patch = AnimaRegionalLoraBlockPatch(
        original,
        execution.attention,
        activation_context=AnimaActivationContext(),
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=AnimaLoraSpatialInvocationContext(),
        schedule_context=AnimaRegionalLoraScheduleContext(),
    )

    assert "original" not in dict(patch.named_modules())
    assert patch.self_attn is original.self_attn
    original_attention = original.self_attn
    patch.self_attn = nn.Identity()
    assert original.self_attn is original_attention


def test_block_blends_complete_adaln_branches_before_one_shared_trajectory() -> None:
    """Apply regional gate modulation while preserving installed stage cardinality."""

    block = _TinyBlock()
    execution = _execution()
    activation = AnimaActivationContext()
    branch = AnimaLoraBranchInvocationContext()
    spatial = AnimaLoraSpatialInvocationContext()
    target = execution.admission.targets[0]
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    block.adaln_modulation_self_attn[2] = AnimaRegionalLoraCompositionLinearPatch(
        block.adaln_modulation_self_attn[2],
        composition.groups_for_target(target.adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=branch,
            spatial_context=spatial,
        ),
        schedule_context=schedule,
    )
    patch = AnimaRegionalLoraBlockPatch(
        block,
        execution.attention,
        activation_context=activation,
        branch_context=branch,
        spatial_context=spatial,
        schedule_context=schedule,
    )
    patch_order: list[str] = []

    def first(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Record and increment the first MLP patch input."""

        patch_order.append("first")
        return {"x": payload["x"] + 1.0}

    def second(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Record and double the second MLP patch input."""

        patch_order.append("second")
        return {"x": payload["x"] * 2.0}

    x = torch.zeros((1, 1, 1, 2, 1))
    embedding = torch.ones((1, 1, 1))
    built_in = torch.tensor([[[0.0, 0.0, 0.25]]])
    options: dict[str, Any] = {"patches": {"mlp_patch": (first, second)}}

    with schedule.activate(resolution), activation.activate(_geometry()):
        output = patch(
            x,
            embedding,
            torch.zeros((1, 1, 1)),
            adaln_lora_B_T_3D=built_in,
            transformer_options=options,
        )

    torch.testing.assert_close(output.flatten(), torch.tensor([2.25, 1.25]))
    assert block.self_attn.calls == 1
    assert block.cross_attn.calls == 1
    assert block.mlp.calls == 1
    assert patch_order == ["first", "second"]
    assert block.mlp.inputs is not None
    torch.testing.assert_close(block.mlp.inputs, (output + 1.0) * 2.0)


@pytest.mark.parametrize(
    ("x", "embedding", "context", "built_in", "message"),
    [
        (
            torch.zeros((1, 1, 1, 1)),
            torch.ones((1, 1, 1)),
            torch.zeros((1, 1, 1)),
            torch.zeros((1, 1, 3)),
            "BxTxHxWxD",
        ),
        (
            torch.zeros((1, 1, 1, 2, 1)),
            torch.ones((1, 2, 1)),
            torch.zeros((1, 1, 1)),
            torch.zeros((1, 1, 3)),
            "batch/time",
        ),
        (
            torch.zeros((1, 1, 1, 2, 1)),
            torch.ones((1, 1, 1)),
            torch.zeros((1, 1, 1)),
            None,
            "built-in",
        ),
    ],
)
def test_block_rejects_invalid_installed_tensor_contracts(
    x: torch.Tensor,
    embedding: torch.Tensor,
    context: torch.Tensor,
    built_in: torch.Tensor | None,
    message: str,
) -> None:
    """Fail before stage work when installed block tensor alignment drifts."""

    execution = _execution()
    activation = AnimaActivationContext()
    composition = AnimaRegionalLoraComposition((execution,))
    schedule, resolution = static_lora_schedule(composition)
    patch = AnimaRegionalLoraBlockPatch(
        _TinyBlock(),
        execution.attention,
        activation_context=activation,
        branch_context=AnimaLoraBranchInvocationContext(),
        spatial_context=AnimaLoraSpatialInvocationContext(),
        schedule_context=schedule,
    )

    with schedule.activate(resolution), activation.activate(_geometry()):
        with pytest.raises((TypeError, ValueError), match=message):
            patch(x, embedding, context, adaln_lora_B_T_3D=built_in)


def _modulation() -> nn.Sequential:
    """Return base shift zero, scale zero, and gate one."""

    first = nn.Linear(1, 1, bias=False)
    second = nn.Linear(1, 3, bias=False)
    with torch.no_grad():
        first.weight.fill_(1.0)
        second.weight.copy_(torch.tensor([[0.0], [0.0], [1.0]]))
    return nn.Sequential(nn.Identity(), first, second)


def _execution() -> AnimaRegionalLoraAdapterExecution:
    """Build one full-rank AdaLN adapter over the first of two query pixels."""

    family = AnimaLoraTargetFamily.ADALN_SELF_ATTN_2
    name = f"diffusion_model.blocks.0.{family.value}"
    adapter = StandardLoraTarget(
        name,
        torch.ones((1, 1)),
        torch.tensor([[0.0], [0.0], [1.0]]),
        rank=1,
        input_features=1,
        output_features=3,
    )
    target = AnimaLoraTarget(0, family, adapter)
    identity = RegionalLoraAdapterIdentity("adapter.safetensors")
    plan = RegionalLoraAdapterPlan(
        identity,
        composition_index=0,
        region_index=0,
        branch=RegionalLoraBranch.POSITIVE,
        model_strength=1.0,
        schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
    )
    context = torch.zeros((1, 1, 1))
    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        context,
        single_entry_regions((context.clone(),)),
    )
    mask = torch.tensor([[[1.0, 0.0]]])
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


def _geometry() -> AnimaActivationGeometry:
    """Return patch-size-one geometry for the two-pixel synthetic block."""

    return AnimaActivationGeometry(1, 1, 1, 2, 1, 1, 1, 1, 2, None)
