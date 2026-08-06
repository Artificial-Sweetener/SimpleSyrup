# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Prompt-Control SEP segment and hook planning."""

from __future__ import annotations

from simple_syrup.services.prompt_control_segment_planning_service import (
    PromptControlSegmentPlanningService,
)


def test_planner_combines_lora_tags_only_at_aligned_indexes() -> None:
    """Positive and negative LoRAs share a plan only within one SEP position."""

    plan = PromptControlSegmentPlanningService().prepare(
        positive_prompt=(
            "global <lora:global-positive:1> [SEP] "
            "left <lora:left-positive:0.5> [SEP] right"
        ),
        negative_prompt=(
            "bad <lora:global-negative:0.25> [SEP] worse <lora:left-negative:[0:1:0.5]>"
        ),
        separator="[SEP]",
    )

    assert plan.is_batched is True
    assert [chunk.text for chunk in plan.positive.chunks] == [
        "global ",
        "left ",
        "right",
    ]
    assert [hook.lora_tags for hook in plan.hooks] == [
        "<lora:global-positive:1>\n<lora:global-negative:0.25>",
        "<lora:left-positive:0.5>\n<lora:left-negative:[0:1:0.5]>",
        "",
    ]


def test_planner_aligns_lora_hooks_across_named_separators() -> None:
    """Treat separator names as comments while aligning segment-local hooks."""

    plan = PromptControlSegmentPlanningService().prepare(
        positive_prompt=(
            "global [SEP|Sky] clouds <lora:sky:1> [SEP|Ground] field <lora:ground:0.5>"
        ),
        negative_prompt="bad [SEP|Sky] haze",
        separator="[SEP]",
    )

    assert [chunk.text for chunk in plan.positive.chunks] == [
        "global",
        "clouds ",
        "field ",
    ]
    assert [chunk.text for chunk in plan.negative.chunks] == ["bad", "haze", "bad"]
    assert [hook.lora_tags for hook in plan.hooks] == [
        "",
        "<lora:sky:1>",
        "<lora:ground:0.5>",
    ]


def test_planner_preserves_empties_and_fills_missing_side_from_global() -> None:
    """Authored empties remain while missing negative positions reuse global text."""

    plan = PromptControlSegmentPlanningService().prepare(
        positive_prompt="global [SEP]  [SEP] right <lora:right:1>",
        negative_prompt="negative",
        separator="[SEP]",
    )

    assert [chunk.text for chunk in plan.positive.chunks] == [
        "global",
        "",
        "right ",
    ]
    assert [chunk.text for chunk in plan.negative.chunks] == [
        "negative",
        "negative",
        "negative",
    ]
    assert [hook.lora_tags for hook in plan.hooks] == ["", "", "<lora:right:1>"]


def test_planner_fallback_text_does_not_repeat_global_lora_tags() -> None:
    """Synthetic segments inherit global text but not global SEP-local tags."""

    plan = PromptControlSegmentPlanningService().prepare(
        positive_prompt="global [SEP] left <lora:left:1> [SEP] right",
        negative_prompt="bad <lora:global-negative:1>",
        separator="[SEP]",
    )

    assert [chunk.text for chunk in plan.negative.chunks] == ["bad ", "bad ", "bad "]
    assert [chunk.lora_tags for chunk in plan.negative.chunks] == [
        "<lora:global-negative:1>",
        "",
        "",
    ]
    assert [hook.lora_tags for hook in plan.hooks] == [
        "<lora:global-negative:1>",
        "<lora:left:1>",
        "",
    ]


def test_planner_fills_missing_positive_side_symmetrically() -> None:
    """Negative-authored regions reuse global positive text with local hooks."""

    plan = PromptControlSegmentPlanningService().prepare(
        positive_prompt="subject <lora:global-positive:1>",
        negative_prompt="bad [SEP] hands <lora:hands:1> [SEP] text",
        separator="[SEP]",
    )

    assert [chunk.text for chunk in plan.positive.chunks] == [
        "subject ",
        "subject ",
        "subject ",
    ]
    assert [chunk.lora_tags for chunk in plan.positive.chunks] == [
        "<lora:global-positive:1>",
        "",
        "",
    ]
    assert [hook.lora_tags for hook in plan.hooks] == [
        "<lora:global-positive:1>",
        "<lora:hands:1>",
        "",
    ]


def test_planner_marks_single_segment_prompts_as_unbatched() -> None:
    """No-SEP prompts retain the compatibility path used by the scheduler."""

    plan = PromptControlSegmentPlanningService().prepare(
        positive_prompt="portrait <lora:portrait:1>",
        negative_prompt="blur",
        separator="[SEP]",
    )

    assert plan.is_batched is False
    assert plan.hooks[0].lora_tags == "<lora:portrait:1>"
