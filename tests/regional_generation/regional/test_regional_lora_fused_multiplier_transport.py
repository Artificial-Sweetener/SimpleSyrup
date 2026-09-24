# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove allocation-free multiplier transport into fused accumulation."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.regional_lora.fused_active_accumulation import (
    RegionalLoraFusedActiveAccumulator,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused multiplier transport requires CUDA.",
)


def test_multiple_adapters_reach_fused_kernel_without_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve ordered math while passing existing multiplier vectors directly."""

    output = torch.zeros((3, 16), device="cuda", dtype=torch.float16)
    rank_values = torch.ones((3, 4, 16), device="cuda", dtype=torch.float16)
    up = torch.ones((4, 16, 16), device="cuda", dtype=torch.float16)
    multipliers = tuple(
        torch.full((3,), float(index + 1), device="cuda", dtype=torch.float16)
        for index in range(4)
    )

    def reject_stack(*_: object, **__: object) -> torch.Tensor:
        raise AssertionError("fused multiplier transport must not stack tensors")

    monkeypatch.setattr(torch, "stack", reject_stack)

    result = RegionalLoraFusedActiveAccumulator().add(
        output,
        rank_values=rank_values,
        up=up,
        multipliers=multipliers,
        indices=None,
    )

    torch.testing.assert_close(result, torch.full_like(result, 160.0))
