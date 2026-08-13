# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify normalized regional attention weighting and output composition."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.regional_attention_weights import (
    RegionalAttentionWeightingPolicy,
)


def test_zero_regions_preserve_the_base_output() -> None:
    """Return the base branch exactly when every regional mask is zero."""

    policy = RegionalAttentionWeightingPolicy()
    masks = torch.zeros((2, 2, 2))
    weights = policy.weights(masks, region_strengths=(1.0, 1.0))
    base = torch.arange(8, dtype=torch.float32).reshape((2, 2, 2))
    regions = torch.stack((base + 10.0, base + 20.0))

    blended = policy.blend(
        weights=weights,
        base_output=base,
        regional_outputs=regions,
    )

    assert torch.equal(weights.base, torch.ones((2, 2)))
    assert torch.equal(weights.regions, masks)
    assert torch.equal(weights.denominator, torch.ones((2, 2)))
    assert torch.equal(blended, base)


def test_partial_and_overlapping_regions_follow_exact_normalized_formula() -> None:
    """Apply base complement and normalize overlaps without changing region order."""

    policy = RegionalAttentionWeightingPolicy()
    masks = torch.tensor([[[0.5, 1.0]], [[0.25, 1.0]]])
    weights = policy.weights(masks, region_strengths=(1.0, 0.5))
    base = torch.tensor([[[10.0], [10.0]]])
    regions = torch.tensor([[[[20.0], [20.0]]], [[[40.0], [40.0]]]])

    blended = policy.blend(
        weights=weights,
        base_output=base,
        regional_outputs=regions,
    )

    expected_region_weights = torch.tensor([[[0.5, 1.0]], [[0.125, 0.5]]])
    expected_base = torch.tensor([[0.375, 0.0]])
    expected_denominator = torch.tensor([[1.0, 1.5]])
    expected = torch.tensor([[[18.75], [80.0 / 3.0]]])
    torch.testing.assert_close(weights.regions, expected_region_weights)
    torch.testing.assert_close(weights.base, expected_base)
    torch.testing.assert_close(weights.denominator, expected_denominator)
    torch.testing.assert_close(blended, expected)
    torch.testing.assert_close(
        weights.normalized_base + weights.normalized_regions.sum(dim=0),
        torch.ones_like(weights.base),
    )


def test_weighting_clamps_masks_and_applies_zero_strength() -> None:
    """Clamp authored coverage before strength while retaining caller tensors."""

    policy = RegionalAttentionWeightingPolicy()
    masks = torch.tensor([[[-1.0, 2.0]], [[1.0, 1.0]]])
    original = masks.to(device=masks.device, dtype=masks.dtype, copy=True)

    weights = policy.weights(masks, region_strengths=(0.5, 0.0))

    assert torch.equal(weights.regions, torch.tensor([[[0.0, 0.5]], [[0.0, 0.0]]]))
    assert torch.equal(weights.base, torch.tensor([[1.0, 0.5]]))
    assert torch.equal(masks, original)


def test_blend_supports_multiple_trailing_feature_dimensions() -> None:
    """Broadcast query weights across arbitrary ordered output feature axes."""

    policy = RegionalAttentionWeightingPolicy()
    weights = policy.weights(torch.ones((1, 2, 3)), region_strengths=(1.0,))
    base = torch.zeros((2, 3, 2, 4))
    regional = torch.full((1, 2, 3, 2, 4), 7.0)

    blended = policy.blend(
        weights=weights,
        base_output=base,
        regional_outputs=regional,
    )

    assert torch.equal(blended, regional[0])


@pytest.mark.parametrize(
    ("masks", "error", "message"),
    [
        (cast(Any, "invalid"), TypeError, "must be a torch.Tensor"),
        (torch.ones((2,)), ValueError, "region and query-grid dimensions"),
        (torch.ones((0, 2)), ValueError, "non-empty region and query grids"),
        (torch.ones((1, 0)), ValueError, "non-empty region and query grids"),
        (
            torch.ones((1, 2), dtype=torch.int64),
            TypeError,
            "floating-point dtype",
        ),
        (torch.tensor([[float("nan")]]), ValueError, "contain finite values"),
        (torch.tensor([[float("inf")]]), ValueError, "contain finite values"),
    ],
)
def test_weighting_rejects_invalid_masks(
    masks: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    """Fail before weight construction for malformed regional masks."""

    with pytest.raises(error, match=message):
        RegionalAttentionWeightingPolicy().weights(
            masks,
            region_strengths=(1.0,),
        )


@pytest.mark.parametrize(
    ("strengths", "error", "message"),
    [
        (cast(Any, [1.0]), TypeError, "immutable tuple"),
        ((1.0,), ValueError, "count must match"),
        ((1.0, True), TypeError, "strength 1 must be a real number"),
        ((1.0, cast(Any, "high")), TypeError, "strength 1 must be a real number"),
        ((1.0, -0.1), ValueError, "strength 1 must be finite and non-negative"),
        (
            (1.0, float("nan")),
            ValueError,
            "strength 1 must be finite and non-negative",
        ),
        (
            (1.0, float("inf")),
            ValueError,
            "strength 1 must be finite and non-negative",
        ),
    ],
)
def test_weighting_rejects_invalid_strengths(
    strengths: tuple[float, ...],
    error: type[Exception],
    message: str,
) -> None:
    """Validate immutable ordered strengths before creating a device tensor."""

    with pytest.raises(error, match=message):
        RegionalAttentionWeightingPolicy().weights(
            torch.ones((2, 1)),
            region_strengths=strengths,
        )


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_weighting_rejects_invalid_epsilon(epsilon: float) -> None:
    """Require a finite positive denominator floor."""

    with pytest.raises(ValueError, match="epsilon must be finite and positive"):
        RegionalAttentionWeightingPolicy().weights(
            torch.ones((1, 1)),
            region_strengths=(1.0,),
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    ("base", "regions", "error", "message"),
    [
        (
            torch.zeros((3, 2)),
            torch.zeros((1, 3, 2)),
            ValueError,
            "must begin with the weighting query grid",
        ),
        (
            torch.zeros((2, 2)),
            torch.zeros((2, 2, 2)),
            ValueError,
            "one ordered branch per region",
        ),
        (
            torch.zeros((2, 2), dtype=torch.float32),
            torch.zeros((1, 2, 2), dtype=torch.float64),
            ValueError,
            "output dtypes must match",
        ),
        (
            torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]),
            torch.zeros((1, 2, 2)),
            ValueError,
            "must contain finite values",
        ),
    ],
)
def test_blend_rejects_invalid_branch_outputs(
    base: torch.Tensor,
    regions: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed for branch output shape, ordering, dtype, or finite errors."""

    weights = RegionalAttentionWeightingPolicy().weights(
        torch.ones((1, 2, 2)),
        region_strengths=(1.0,),
    )
    with pytest.raises(error, match=message):
        RegionalAttentionWeightingPolicy().blend(
            weights=weights,
            base_output=base,
            regional_outputs=regions,
        )
