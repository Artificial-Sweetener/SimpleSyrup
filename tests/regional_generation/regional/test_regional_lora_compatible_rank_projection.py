# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove reusable compatible rank projections preserve exact active math."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.regional_lora.compatible_rank_projection import (
    RegionalLoraCompatibleRankProjector,
)


def test_compatible_projection_uses_dense_a_projection_at_half_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch adapter ranks densely when gathering would save only half the rows."""

    inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    down = torch.cat((torch.eye(2), torch.eye(2)), dim=0)
    up = torch.stack((torch.eye(2), torch.eye(2)))
    multipliers = (
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.5, 0.0]),
    )
    observed_batches: list[int] = []
    original_linear = torch.nn.functional.linear

    def recording_linear(
        values: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record the one exact shared dense A projection."""

        observed_batches.append(int(values.shape[0]))
        return original_linear(values, weights, bias)

    monkeypatch.setattr(
        "simple_syrup.runtime.regional_lora.compatible_rank_projection.functional.linear",
        recording_linear,
    )
    projector = RegionalLoraCompatibleRankProjector()
    projection = projector.prepare(
        inputs,
        down=down,
        up=up,
        multipliers=multipliers,
        rank=2,
    )
    deltas = projector.materialize_deltas(projection)

    expected = tuple(inputs * multiplier[:, None] for multiplier in multipliers)
    assert len(deltas) == len(expected)
    for observed, expected_delta in zip(deltas, expected, strict=True):
        torch.testing.assert_close(observed, expected_delta)
        assert observed.is_contiguous()
    assert observed_batches == [4]


def test_materialized_deltas_stay_contiguous_with_multi_axis_inputs() -> None:
    """Keep every target contiguous across real transformer leading dimensions."""

    inputs = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)
    down = torch.cat((torch.eye(2, 8), torch.eye(2, 8)), dim=0)
    up = torch.stack((torch.eye(4, 2), torch.eye(4, 2)))
    multipliers = (
        torch.ones((2, 3), dtype=torch.float32),
        torch.full((2, 3), 0.5, dtype=torch.float32),
    )
    projector = RegionalLoraCompatibleRankProjector()
    projection = projector.prepare(
        inputs,
        down=down,
        up=up,
        multipliers=multipliers,
        rank=2,
    )

    deltas = projector.materialize_deltas(projection)

    assert tuple(delta.shape for delta in deltas) == ((2, 3, 4), (2, 3, 4))
    assert all(delta.is_contiguous() for delta in deltas)


@pytest.mark.parametrize("sparse", [False, True])
def test_each_target_accumulates_from_one_shared_rank_projection(
    sparse: bool,
) -> None:
    """Preserve nonzero base outputs without materializing complete deltas."""

    inputs = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75], [0.5, 0.25]])
    down_parts = (torch.tensor([[1.0, 0.5]]), torch.tensor([[-0.25, 1.0]]))
    up_parts = (torch.tensor([[0.75], [1.5]]), torch.tensor([[1.0], [-0.5]]))
    down = torch.cat(down_parts)
    up = torch.stack(up_parts)
    first_multiplier = torch.tensor([1.0, 0.5, 0.25, 0.75])
    second_multiplier = torch.tensor([0.5, 0.25, 1.0, 0.75])
    if sparse:
        first_multiplier[1::2] = 0.0
        second_multiplier[1::2] = 0.0
    multipliers = (first_multiplier, second_multiplier)
    projector = RegionalLoraCompatibleRankProjector()
    projection = projector.prepare(
        inputs,
        down=down,
        up=up,
        multipliers=multipliers,
        rank=1,
    )
    first_base = torch.full((4, 2), 3.0)
    second_base = torch.full((4, 2), -2.0)

    first = projector.add_target(first_base.clone(), projection, 0)
    second = projector.add_target(second_base.clone(), projection, 1)

    expected_first = (
        first_base
        + ((inputs @ down_parts[0].T) * first_multiplier[:, None]) @ up_parts[0].T
    )
    expected_second = (
        second_base
        + ((inputs @ down_parts[1].T) * second_multiplier[:, None]) @ up_parts[1].T
    )
    torch.testing.assert_close(first, expected_first)
    torch.testing.assert_close(second, expected_second)
