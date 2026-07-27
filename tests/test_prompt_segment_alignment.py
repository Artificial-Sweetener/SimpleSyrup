# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for authored and fallback SEP segment alignment."""

from __future__ import annotations

import pytest

from simple_syrup.domain.prompt_segment_alignment import (
    build_prompt_segment_alignment,
)


def test_alignment_reuses_global_value_for_missing_negative_segments() -> None:
    """Missing negative positions resolve to the authored global entry."""

    plan = build_prompt_segment_alignment(positive_count=3, negative_count=1)

    assert plan.segment_count == 3
    assert plan.positive.materialize(("global", "left", "right")) == (
        "global",
        "left",
        "right",
    )
    assert plan.negative.materialize(("bad",)) == ("bad", "bad", "bad")
    assert [source.authored for source in plan.negative.sources] == [True, False, False]
    assert [source.source_index for source in plan.negative.sources] == [0, 0, 0]


def test_alignment_is_symmetric_for_missing_positive_segments() -> None:
    """Missing positive positions use global positive rather than a prior region."""

    plan = build_prompt_segment_alignment(positive_count=1, negative_count=3)

    assert plan.positive.materialize(("global",)) == (
        "global",
        "global",
        "global",
    )
    assert plan.negative.materialize(("bad", "hands", "text")) == (
        "bad",
        "hands",
        "text",
    )


def test_alignment_preserves_authored_empty_segments() -> None:
    """An authored empty position remains empty instead of falling back."""

    plan = build_prompt_segment_alignment(positive_count=3, negative_count=1)

    assert plan.positive.materialize(("global", "", "right")) == (
        "global",
        "",
        "right",
    )
    assert plan.positive.sources[1].authored is True


@pytest.mark.parametrize(
    ("positive_count", "negative_count"),
    [(0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_alignment_rejects_sides_without_a_global_segment(
    positive_count: int,
    negative_count: int,
) -> None:
    """Each side must expose index 0 before fallback can be planned."""

    with pytest.raises(ValueError, match="at least one authored segment"):
        build_prompt_segment_alignment(
            positive_count=positive_count,
            negative_count=negative_count,
        )
