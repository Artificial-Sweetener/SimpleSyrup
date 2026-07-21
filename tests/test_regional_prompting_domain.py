# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for global-first regional prompt pairing policy."""

from __future__ import annotations

import pytest

from simple_syrup.domain.regional_prompting import (
    RegionalConditioningPair,
    build_regional_conditioning_plan,
    validate_regional_prompt_weight,
)


def test_pairs_every_available_regional_entry_in_order() -> None:
    """Entry zero stays global and later entries map directly to masks."""

    plan = build_regional_conditioning_plan(
        region_count=3,
        conditioning_count=3,
        input_name="positive",
    )

    assert plan.region_count == 3
    assert plan.pairs == (
        RegionalConditioningPair(conditioning_index=1, mask_index=0),
        RegionalConditioningPair(conditioning_index=2, mask_index=1),
    )


def test_fewer_regional_entries_than_masks_is_valid() -> None:
    """Unpaired authored masks intentionally retain global conditioning only."""

    plan = build_regional_conditioning_plan(
        region_count=4,
        conditioning_count=1,
        input_name="negative",
    )

    assert plan.pairs == ()


def test_more_regional_entries_than_masks_fails_actionably() -> None:
    """Excess prompts cannot silently lose their positional meaning."""

    with pytest.raises(
        ValueError,
        match="positive conditioning contains 2 regional entries but only 1",
    ):
        build_regional_conditioning_plan(
            region_count=1,
            conditioning_count=3,
            input_name="positive",
        )


@pytest.mark.parametrize(
    ("region_count", "conditioning_count", "message"),
    [
        (0, 1, "at least one authored mask"),
        (1, 0, "global entry at index 0"),
    ],
)
def test_invalid_plan_counts_fail(
    region_count: int,
    conditioning_count: int,
    message: str,
) -> None:
    """A plan always contains masks and a structural global entry."""

    with pytest.raises(ValueError, match=message):
        build_regional_conditioning_plan(
            region_count=region_count,
            conditioning_count=conditioning_count,
            input_name="positive",
        )


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_regional_prompt_weight_accepts_normalized_range(weight: float) -> None:
    """Regional influence accepts both endpoints and the default midpoint."""

    validate_regional_prompt_weight(weight)


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("inf"), float("nan")])
def test_regional_prompt_weight_rejects_invalid_values(weight: float) -> None:
    """Invalid regional influence fails before conditioning assembly."""

    with pytest.raises(ValueError, match="regional_prompt_weight"):
        validate_regional_prompt_weight(weight)
