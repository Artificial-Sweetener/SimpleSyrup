# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify Anima global prompt sharing throughout active regions."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.regional_lora.anima_cross_attention_weights import (
    ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY,
    AnimaCrossAttentionWeightingPolicy,
)


def test_base_only_attention_remains_exact() -> None:
    """Preserve the ordinary global result when no regional mask is active."""

    policy = ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY
    weights = policy.weights(
        torch.zeros((2, 1, 4)),
        region_strengths=(1.0, 1.0),
    )

    assert torch.equal(weights.base, torch.ones((1, 4)))
    assert torch.equal(weights.regions, torch.zeros((2, 1, 4)))


def test_solid_region_retains_fixed_global_composition_share() -> None:
    """Keep the scene prompt active while regional text remains authoritative."""

    weights = ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY.weights(
        torch.ones((1, 1, 64)),
        region_strengths=(1.0,),
    )

    torch.testing.assert_close(weights.base, torch.full((1, 64), 1.0 / 3.0))
    torch.testing.assert_close(weights.regions, torch.full((1, 1, 64), 2.0 / 3.0))


def test_hard_partition_retains_global_prompt_across_both_regions() -> None:
    """Coordinate one scene across the full authored subject partition."""

    masks = torch.zeros((2, 1, 256))
    grid = masks.reshape(2, 1, 16, 16)
    grid[0, :, :, :8] = 1.0
    grid[1, :, :, 8:] = 1.0

    weights = ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY.weights(
        masks,
        region_strengths=(1.0, 1.0),
    )
    base = weights.base.reshape(1, 16, 16)

    torch.testing.assert_close(base, torch.full((1, 16, 16), 1.0 / 3.0))
    torch.testing.assert_close(
        weights.normalized_base + weights.normalized_regions.sum(dim=0),
        torch.ones_like(weights.base),
    )


def test_direct_overlap_preserves_fixed_global_share() -> None:
    """Preserve global coordination where authored masks overlap directly."""

    weights = ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY.weights(
        torch.ones((2, 1, 1)),
        region_strengths=(1.0, 1.0),
    )

    torch.testing.assert_close(weights.base, torch.tensor([[2.0 / 3.0]]))
    torch.testing.assert_close(weights.regions, torch.full((2, 1, 1), 2.0 / 3.0))
    torch.testing.assert_close(weights.denominator, torch.tensor([[2.0]]))


@pytest.mark.parametrize("share", [True, 0.0, 1.0, float("nan"), float("inf")])
def test_policy_rejects_invalid_global_shares(share: float) -> None:
    """Fail closed when the Anima composition share is not fractional."""

    with pytest.raises(ValueError, match="between zero and one"):
        AnimaCrossAttentionWeightingPolicy(share)
