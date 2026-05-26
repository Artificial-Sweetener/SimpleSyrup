# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Prompt-Control prompt preparation helpers."""

from __future__ import annotations

import pytest

from simple_syrup.domain.prompt_control_prompt import (
    apply_encode_style,
    extract_lora_tags,
    extract_prompt_text,
    prepare_prompt_side,
)


def test_extract_prompt_text_matches_sugarcubes_regex_behavior() -> None:
    """Visible text outside angle tags is retained and joined with newlines."""

    text = "portrait <lora:face:1.0> cinematic <foo>"

    assert extract_prompt_text(text) == "portrait \n cinematic "


def test_extract_prompt_text_keeps_text_after_adjacent_tag() -> None:
    """Text after a closing angle tag is captured as a later prompt segment."""

    assert extract_prompt_text("<lora:a:1.0>face") == "face"


def test_extract_lora_tags_returns_angle_tags_joined_with_newlines() -> None:
    """All Prompt-Control angle tags are retained for LoRA scheduling."""

    text = "portrait <lora:face:1.0> cinematic <lora:light:0.5>"

    assert extract_lora_tags(text) == "<lora:face:1.0>\n<lora:light:0.5>"


def test_prepare_prompt_side_splits_ordered_chunks_and_aggregates_loras() -> None:
    """Separator-delimited chunks preserve order and collect all tags."""

    side = prepare_prompt_side(
        "face <lora:a:1.0> [SEP] hair <lora:b:0.5>",
        "[SEP]",
    )

    assert [chunk.text for chunk in side.chunks] == ["face ", "hair "]
    assert [chunk.lora_tags for chunk in side.chunks] == [
        "<lora:a:1.0>",
        "<lora:b:0.5>",
    ]
    assert side.lora_tags == "<lora:a:1.0>\n<lora:b:0.5>"


def test_prepare_prompt_side_preserves_empty_chunks() -> None:
    """Batch splitting keeps empty entries so batch positions stay explicit."""

    side = prepare_prompt_side("face [SEP] ", "[SEP]")

    assert [chunk.text for chunk in side.chunks] == ["face", ""]
    assert [chunk.lora_tags for chunk in side.chunks] == ["", ""]
    assert side.lora_tags == ""


def test_prepare_prompt_side_rejects_empty_separator() -> None:
    """Empty separators are rejected by the shared batch splitter."""

    with pytest.raises(ValueError, match="separator must not be empty"):
        prepare_prompt_side("face", "")


def test_apply_encode_style_prepends_style_without_extra_formatting() -> None:
    """Encode style text is used exactly as produced by style nodes."""

    assert apply_encode_style("STYLE(A1111) ", "face") == "STYLE(A1111) face"


def test_apply_encode_style_keeps_prompt_when_style_is_blank() -> None:
    """Blank encode style leaves cleaned prompt text unchanged."""

    assert apply_encode_style("", "face") == "face"
