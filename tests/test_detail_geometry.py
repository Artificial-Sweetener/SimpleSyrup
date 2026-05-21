# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for scale-factor detail geometry."""

from __future__ import annotations

import pytest

from simple_syrup.domain.detail_geometry import (
    DetailScalePlan,
    build_detail_scale_plan,
)


def test_clamp_zero_leaves_scaled_size_unclamped() -> None:
    """A clamp of zero means no maximum crop size."""

    plan = build_detail_scale_plan(20, 10, 100, 50, 1.5, 0)

    assert plan == DetailScalePlan(
        width=150,
        height=75,
        scale=1.5,
        unclamped_long_side=150.0,
        target_long_side=150.0,
    )


def test_positive_clamp_caps_long_side() -> None:
    """A positive clamp limits the scaled crop long side."""

    plan = build_detail_scale_plan(20, 10, 100, 50, 2.0, 120)

    assert plan.width == 120
    assert plan.height == 60
    assert plan.scale == 1.2


def test_scale_uses_crop_size_not_raw_detection_size() -> None:
    """The crop/detail region controls scale-factor geometry."""

    plan = build_detail_scale_plan(10, 10, 100, 50, 2.0, 0)

    assert plan.width == 200
    assert plan.height == 100


def test_small_regions_do_not_produce_zero_size() -> None:
    """Tiny values still produce usable pixel dimensions."""

    plan = build_detail_scale_plan(1, 1, 1, 1, 0.1, 0)

    assert plan.width == 1
    assert plan.height == 1


@pytest.mark.parametrize(
    ("scale_factor", "clamp_size", "message"),
    [
        (0.0, 0, "scale_factor must be greater than 0"),
        (1.0, -1, "clamp_size must be 0 or greater"),
    ],
)
def test_invalid_scale_settings_raise_value_error(
    scale_factor: float,
    clamp_size: int,
    message: str,
) -> None:
    """Invalid scale settings fail before detail work begins."""

    with pytest.raises(ValueError, match=message):
        build_detail_scale_plan(1, 1, 8, 8, scale_factor, clamp_size)
