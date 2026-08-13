# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify live standard-UNet regional operation session resolution."""

from __future__ import annotations

from uuid import UUID

import pytest
import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.linear_execution_plan import (
    RegionalLinearExecutionPlan,
    RegionalLinearOperationKey,
    RegionalLinearTargetUse,
)
from simple_syrup.runtime.regional_lora.preparation import (
    RegionalLoraTargetPreparation,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget
from simple_syrup.runtime.regional_lora.standard_unet_operation_session import (
    StandardUnetRegionalOperationSession,
)
from simple_syrup.runtime.regional_lora.target_binding import (
    BoundRegionalLoraSpatialCapability,
)


def test_session_resolves_live_spatial_geometry_cfg_and_schedule_multiplier() -> None:
    """Project one lower-resolution grid and gate it to the positive CFG row."""

    adapter = _adapter()
    session = StandardUnetRegionalOperationSession(
        RegionalLoraPlan((adapter,)),
        _mask_bank(),
        {"diffusion_model.linear": BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS},
    )
    contexts = _contexts()
    inputs = torch.zeros((2, 2, 2))
    options: dict[str, object] = {
        "sample_sigmas": torch.tensor([2.0, 1.0, 0.0]),
        "sigmas": torch.tensor([1.0, 1.0]),
        "activations_shape": [2, 4, 1, 2],
    }

    with session.activate(contexts, options):
        invocation = session.resolve(
            "diffusion_model.linear",
            _linear_plan(adapter),
            inputs,
        )

    assert invocation is not None
    assert invocation.schedule_strengths == (0.25,)
    assert invocation.masks.geometry.spatial_height == 1
    assert invocation.masks.geometry.spatial_width == 2
    torch.testing.assert_close(
        invocation.masks.multipliers[:, :, :, 0],
        torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
    )
    with pytest.raises(RuntimeError, match="outside a call"):
        session.resolve("diffusion_model.linear", _linear_plan(adapter), inputs)


def _adapter() -> RegionalLoraAdapterPlan:
    """Return one scheduled positive regional adapter plan."""

    return RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity("adapter.safetensors"),
        0,
        0,
        RegionalLoraBranch.POSITIVE,
        0.5,
        (RegionalLoraScheduleBoundary(0.0, 2.0, 0.25, 0),),
    )


def _linear_plan(adapter: RegionalLoraAdapterPlan) -> RegionalLinearExecutionPlan:
    """Return one two-feature identity operation owned by the adapter."""

    target = StandardLoraTarget(
        "diffusion_model.linear.weight",
        torch.eye(2),
        torch.eye(2),
        2,
        2,
        2,
        1.0,
    )
    preparation = RegionalLoraTargetPreparation(
        adapter.adapter_identity,
        ModelCloneLineage(UUID(int=1), UUID(int=2)),
        target,
        RegionalLoraExecutionCache(),
    )
    return RegionalLinearExecutionPlan(
        (
            RegionalLinearTargetUse(
                0,
                0,
                RegionalLoraBranch.POSITIVE,
                RegionalLinearOperationKey(
                    adapter.adapter_identity,
                    target.target,
                    id(target.down),
                    id(target.up),
                ),
                preparation,
                0.5,
            ),
        )
    )


def _contexts() -> BatchedRegionalAttentionContexts:
    """Return positive then negative CFG rows with one region."""

    base = torch.zeros((2, 77, 4))
    return BatchedRegionalAttentionContexts(
        1,
        (
            RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.NEGATIVE, 1, 2),
        ),
        base,
        (
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, base.clone(), (1.0, 1.0)),),
            ),
        ),
    )


def _mask_bank() -> RegionalMaskBank:
    """Return a left-half mask authored at a larger four-token canvas."""

    mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    return RegionalMaskBank(mask, mask.clone(), 4, 1)
