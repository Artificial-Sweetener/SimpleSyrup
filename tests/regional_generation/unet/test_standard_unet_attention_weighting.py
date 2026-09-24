# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize explicit global-plus-regional standard-UNet weighting."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.attention_coupling.standard_unet_attention_weighting import (
    StandardUnetAttentionWeightingPolicy,
)


def test_standard_weighting_uses_prevalidated_tensor_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid device scalar validation after canonical mask admission."""

    def reject_redundant_finiteness(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("standard weighting repeated tensor-value validation")

    monkeypatch.setattr(torch, "isfinite", reject_redundant_finiteness)
    policy = StandardUnetAttentionWeightingPolicy()
    weights = policy.weights(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        region_strengths=(0.4, 0.4),
    )

    result = policy.blend(
        weights=weights,
        base_output=torch.tensor([[10.0], [20.0]]),
        regional_outputs=torch.tensor([[[30.0], [40.0]], [[50.0], [60.0]]]),
    )

    torch.testing.assert_close(result, torch.tensor([[18.0], [36.0]]))


def test_standard_weighting_retains_global_branch_under_single_region() -> None:
    """Match the accepted 0.6 global plus 0.4 local PPM balance."""

    weights = StandardUnetAttentionWeightingPolicy().weights(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        region_strengths=(0.4, 0.4),
    )

    torch.testing.assert_close(weights.normalized_base, torch.tensor([0.6, 0.6]))
    torch.testing.assert_close(
        weights.normalized_regions,
        torch.tensor([[0.4, 0.0], [0.0, 0.4]]),
    )


def test_standard_weighting_normalizes_overlap_with_explicit_global() -> None:
    """Keep one global branch active while both regional branches overlap."""

    weights = StandardUnetAttentionWeightingPolicy().weights(
        torch.ones((2, 1)),
        region_strengths=(0.4, 0.4),
    )

    torch.testing.assert_close(weights.normalized_base, torch.tensor([3.0 / 7.0]))
    torch.testing.assert_close(
        weights.normalized_regions,
        torch.tensor([[2.0 / 7.0], [2.0 / 7.0]]),
    )


def test_standard_weighting_keeps_uncovered_and_negative_rows_global() -> None:
    """Normalize zero regional coverage to exact base behavior."""

    weights = StandardUnetAttentionWeightingPolicy().weights(
        torch.zeros((2, 3)),
        region_strengths=(0.4, 0.4),
    )

    torch.testing.assert_close(weights.normalized_base, torch.ones(3))
    torch.testing.assert_close(weights.normalized_regions, torch.zeros((2, 3)))


def test_standard_weighting_preserves_region_owned_endpoint() -> None:
    """Keep public regional strength one as an exact zero-global endpoint."""

    weights = StandardUnetAttentionWeightingPolicy().weights(
        torch.tensor([[0.25], [0.0]]),
        region_strengths=(1.0, 1.0),
    )

    torch.testing.assert_close(weights.normalized_base, torch.zeros(1))
    torch.testing.assert_close(
        weights.normalized_regions,
        torch.tensor([[1.0], [0.0]]),
    )


def test_standard_weighting_keeps_endpoint_uncovered_rows_global() -> None:
    """Retain exact base behavior where no regional branch has support."""

    weights = StandardUnetAttentionWeightingPolicy().weights(
        torch.zeros((2, 1)),
        region_strengths=(1.0, 1.0),
    )

    torch.testing.assert_close(weights.normalized_base, torch.ones(1))
    torch.testing.assert_close(weights.normalized_regions, torch.zeros((2, 1)))
