# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify immutable unique-target mapping for regional Linear fusion."""

from __future__ import annotations

from uuid import UUID

import torch

from simple_syrup.domain.regional_lora_plan import RegionalLoraAdapterIdentity
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.linear_mapped_projection_plan import (
    RegionalLinearMappedProjectionPlan,
)
from simple_syrup.runtime.regional_lora.preparation import RegionalLoraTargetPreparation
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


def test_repeated_exact_targets_map_to_one_compatible_preparation() -> None:
    """Intern exact A/B tensor identities while retaining declared group order."""

    first = _preparation("first", rank=2, offset=0.0)
    second = _preparation("second", rank=2, offset=0.25)

    plan = RegionalLinearMappedProjectionPlan((first, second, first, second))

    assert plan.supported
    assert plan.unique_preparations == (first, second)
    assert plan.group_target_indices == (0, 1, 0, 1)
    first_indices = plan.target_indices(torch.device("cpu"))
    assert first_indices.tolist() == [0, 1, 0, 1]
    assert plan.target_indices(torch.device("cpu")) is first_indices
    plan.clear()
    assert plan._device_indices == {}


def test_mixed_ranks_are_zero_padded_without_truncation() -> None:
    """Fuse distinct full ranks through one maximum-rank storage contract."""

    plan = RegionalLinearMappedProjectionPlan(
        (
            _preparation("first", rank=1, offset=0.0),
            _preparation("second", rank=2, offset=0.25),
        )
    )

    weights = plan.weights(device=torch.device("cpu"), dtype=torch.float32)

    assert plan.supported
    assert plan.rank == 2
    assert tuple(weights.down.shape) == (4, 4)
    assert tuple(weights.up.shape) == (2, 3, 2)
    torch.testing.assert_close(weights.down[1], torch.zeros(4))
    torch.testing.assert_close(weights.up[0, :, 1], torch.zeros(3))


def test_incompatible_feature_shapes_decline_mapped_projection() -> None:
    """Leave mixed feature contracts to exact partitioned execution."""

    plan = RegionalLinearMappedProjectionPlan(
        (
            _preparation("first", rank=1, offset=0.0),
            _preparation("second", rank=2, offset=0.25, input_features=5),
        )
    )

    assert not plan.supported
    assert plan.group_target_indices == ()


def _preparation(
    name: str,
    *,
    rank: int,
    offset: float,
    input_features: int = 4,
) -> RegionalLoraTargetPreparation:
    """Return one generic prepared target with stable tensor identities."""

    down = (
        torch.arange(rank * input_features, dtype=torch.float32).reshape(
            rank,
            input_features,
        )
        + offset
    )
    up = torch.arange(3 * rank, dtype=torch.float32).reshape(3, rank) + offset
    return RegionalLoraTargetPreparation(
        RegionalLoraAdapterIdentity(name),
        ModelCloneLineage(UUID(int=1), UUID(int=2)),
        StandardLoraTarget(name, down, up, rank, input_features, 3, 1.0),
        RegionalLoraExecutionCache(),
    )
