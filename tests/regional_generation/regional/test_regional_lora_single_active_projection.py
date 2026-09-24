# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove one regional-LoRA target excludes exact zero-support positions."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.regional_lora.single_active_projection import (
    RegionalLoraSingleActiveProjectionExecutor,
)


def test_single_projection_gathers_only_nonzero_rows_and_restores_zeros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run both low-rank kernels on active rows and scatter exact full shape."""

    inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    down = torch.eye(2)
    up = torch.eye(2)
    multiplier = torch.tensor([1.0, 0.0, 0.0, 0.0])
    observed_batches: list[int] = []
    original_linear = torch.nn.functional.linear

    def recording_linear(
        values: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record each exact compact projection batch."""

        observed_batches.append(int(values.shape[0]))
        return original_linear(values, weights, bias)

    monkeypatch.setattr(
        "simple_syrup.runtime.regional_lora.single_active_projection.functional.linear",
        recording_linear,
    )
    delta = RegionalLoraSingleActiveProjectionExecutor().delta(
        inputs,
        down=down,
        up=up,
        multiplier=multiplier,
    )

    expected = inputs * multiplier[:, None]
    torch.testing.assert_close(delta, expected)
    assert observed_batches == [1, 1]
