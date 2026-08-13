# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove immutable authored and converted conditioning schedule contracts."""

from __future__ import annotations

import pytest

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((None, None, None, None), (None, None, None, None)),
        ((0, 1, 100, 0), (0.0, 1.0, 100.0, 0.0)),
        ((0.0, 0.5, 100.0, 50.0), (0.0, 0.5, 100.0, 50.0)),
        ((0.25, 0.75, 75.0, 25.0), (0.25, 0.75, 75.0, 25.0)),
        ((0.5, 0.5, 50.0, 50.0), (0.5, 0.5, 50.0, 50.0)),
    ],
)
def test_schedule_retains_unbounded_full_adjacent_overlap_and_zero_width_ranges(
    values: tuple[float | int | None, ...],
    expected: tuple[float | None, ...],
) -> None:
    """Normalize exact finite values while preserving absent boundaries."""

    schedule = ConditioningScheduleRange(*values)

    assert (
        schedule.start_percent,
        schedule.end_percent,
        schedule.timestep_start,
        schedule.timestep_end,
    ) == expected


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ((True, None, None, None), TypeError, "start_percent"),
        (("0", None, None, None), TypeError, "start_percent"),
        ((float("nan"), None, None, None), ValueError, "finite"),
        ((-0.01, None, None, None), ValueError, r"\[0, 1\]"),
        ((None, 1.01, None, None), ValueError, r"\[0, 1\]"),
        ((0.75, 0.25, None, None), ValueError, "must not exceed"),
        ((None, None, float("inf"), None), ValueError, "finite"),
        ((None, None, 25.0, 75.0), ValueError, "must not be below"),
    ],
)
def test_schedule_rejects_malformed_or_reversed_boundaries(
    values: tuple[object, object, object, object],
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed for ambiguous authored or converted schedule state."""

    with pytest.raises(error, match=message):
        ConditioningScheduleRange(*values)  # type: ignore[arg-type]
