# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for pure resize geometry planning."""

from __future__ import annotations

import pytest

from simple_syrup.image.resize_geometry import (
    CropPosition,
    ResizeMode,
    ResizePlan,
    ResizeTarget,
    apply_divisibility,
    build_resize_plan,
    calculate_crop_offsets,
    calculate_pad_sides,
    cover_bounds,
    fit_inside,
)


def test_fit_inside_preserves_aspect_within_bounds() -> None:
    """Fit geometry keeps aspect ratio inside target bounds."""

    assert fit_inside(800, 400, 300, 300) == (300, 150)


def test_cover_bounds_preserves_aspect_while_covering_target() -> None:
    """Cover geometry scales until both target dimensions are covered."""

    assert cover_bounds(800, 400, 300, 300) == (600, 300)


def test_stretch_plan_uses_target_dimensions() -> None:
    """Stretch mode resizes directly to the target dimensions."""

    plan = build_resize_plan(
        800,
        400,
        ResizeTarget(width=300, height=200, divisible_by=1),
        ResizeMode.STRETCH,
        CropPosition.CENTER,
    )

    assert plan == ResizePlan(
        resize_width=300,
        resize_height=200,
        output_width=300,
        output_height=200,
    )


def test_keep_ar_plan_outputs_fitted_dimensions() -> None:
    """Keep AR mode outputs the fitted size instead of padding."""

    plan = build_resize_plan(
        800,
        400,
        ResizeTarget(width=300, height=300, divisible_by=1),
        ResizeMode.KEEP_AR,
        CropPosition.CENTER,
    )

    assert plan == ResizePlan(
        resize_width=300,
        resize_height=150,
        output_width=300,
        output_height=150,
    )


def test_crop_plan_cover_scales_and_offsets() -> None:
    """Crop mode cover-scales and crops according to the chosen anchor."""

    plan = build_resize_plan(
        800,
        400,
        ResizeTarget(width=300, height=300, divisible_by=1),
        ResizeMode.CROP,
        CropPosition.RIGHT,
    )

    assert plan == ResizePlan(
        resize_width=600,
        resize_height=300,
        output_width=300,
        output_height=300,
        crop_x=300,
        crop_y=0,
    )


def test_pad_plan_fit_scales_and_offsets() -> None:
    """Pad mode fit-scales and pads according to the chosen anchor."""

    plan = build_resize_plan(
        800,
        400,
        ResizeTarget(width=300, height=300, divisible_by=1),
        ResizeMode.PAD,
        CropPosition.BOTTOM,
    )

    assert plan == ResizePlan(
        resize_width=300,
        resize_height=150,
        output_width=300,
        output_height=300,
        pad_left=0,
        pad_right=0,
        pad_top=150,
        pad_bottom=0,
    )


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (CropPosition.TOP_LEFT, (0, 0)),
        (CropPosition.TOP, (50, 0)),
        (CropPosition.TOP_RIGHT, (100, 0)),
        (CropPosition.LEFT, (0, 25)),
        (CropPosition.CENTER, (50, 25)),
        (CropPosition.RIGHT, (100, 25)),
        (CropPosition.BOTTOM_LEFT, (0, 50)),
        (CropPosition.BOTTOM, (50, 50)),
        (CropPosition.BOTTOM_RIGHT, (100, 50)),
    ],
)
def test_crop_offsets_are_deterministic(
    position: CropPosition,
    expected: tuple[int, int],
) -> None:
    """All crop positions map to deterministic offsets."""

    assert calculate_crop_offsets(position, 300, 200, 200, 150) == expected


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (CropPosition.TOP_LEFT, (0, 100, 0, 50)),
        (CropPosition.TOP, (50, 50, 0, 50)),
        (CropPosition.TOP_RIGHT, (100, 0, 0, 50)),
        (CropPosition.LEFT, (0, 100, 25, 25)),
        (CropPosition.CENTER, (50, 50, 25, 25)),
        (CropPosition.RIGHT, (100, 0, 25, 25)),
        (CropPosition.BOTTOM_LEFT, (0, 100, 50, 0)),
        (CropPosition.BOTTOM, (50, 50, 50, 0)),
        (CropPosition.BOTTOM_RIGHT, (100, 0, 50, 0)),
    ],
)
def test_pad_sides_are_deterministic(
    position: CropPosition,
    expected: tuple[int, int, int, int],
) -> None:
    """All pad positions map to deterministic side values."""

    assert calculate_pad_sides(position, 100, 50) == expected


def test_divisibility_steps_dimensions_down() -> None:
    """Divisibility floors dimensions to the nearest positive multiple."""

    assert apply_divisibility(1025, 769, 64) == (1024, 768)


def test_divisibility_applies_to_stretch_plan() -> None:
    """Stretch plan applies divisibility to final dimensions."""

    plan = build_resize_plan(
        800,
        400,
        ResizeTarget(width=1025, height=769, divisible_by=64),
        ResizeMode.STRETCH,
        CropPosition.CENTER,
    )

    assert (plan.output_width, plan.output_height) == (1024, 768)


def test_invalid_dimensions_raise_value_error() -> None:
    """Invalid dimensions produce actionable errors."""

    with pytest.raises(ValueError, match="source_width must be greater than 0"):
        build_resize_plan(
            0,
            400,
            ResizeTarget(width=300, height=300, divisible_by=1),
            ResizeMode.KEEP_AR,
            CropPosition.CENTER,
        )


def test_impossible_divisibility_raises_value_error() -> None:
    """Dimensions smaller than divisibility cannot produce valid output."""

    with pytest.raises(ValueError, match="cannot satisfy divisible_by=64"):
        apply_divisibility(32, 128, 64)


def test_invalid_mode_raises_value_error() -> None:
    """Unknown resize modes are rejected explicitly."""

    with pytest.raises(ValueError, match="Unsupported resize mode"):
        build_resize_plan(
            800,
            400,
            ResizeTarget(width=300, height=300, divisible_by=1),
            "bad mode",
            CropPosition.CENTER,
        )
