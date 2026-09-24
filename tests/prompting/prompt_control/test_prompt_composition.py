# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for prompt text composition helpers."""

from __future__ import annotations

from simple_syrup.domain.prompt_composition import prefix_prompt


def test_prefix_prompt_blank_prefix_preserves_prompt() -> None:
    """Blank prefix text leaves the prompt text unchanged."""

    assert prefix_prompt("", "blue eyes") == "blue eyes"


def test_prefix_prompt_non_blank_prefix_prepends_with_comma_space() -> None:
    """Non-blank prefix text is prepended to non-blank prompt text."""

    assert prefix_prompt("masterpiece", "blue eyes") == "masterpiece, blue eyes"


def test_prefix_prompt_non_blank_prefix_with_blank_prompt_returns_prefix() -> None:
    """Blank prompt text does not add a separator after the prefix."""

    assert prefix_prompt("masterpiece", "") == "masterpiece"


def test_prefix_prompt_both_blank_returns_blank() -> None:
    """Blank prefix and prompt text compose to blank text."""

    assert prefix_prompt("", "") == ""


def test_prefix_prompt_treats_whitespace_only_values_as_blank() -> None:
    """Whitespace-only prefix and prompt values behave like blank values."""

    assert prefix_prompt("   ", "  blue eyes  ") == "blue eyes"
    assert prefix_prompt("  masterpiece  ", "   ") == "masterpiece"


def test_prefix_prompt_strips_edges_and_preserves_internal_commas() -> None:
    """Composition trims edges while leaving prompt punctuation alone."""

    assert (
        prefix_prompt(" high detail, best quality ", " blue eyes, looking at viewer ")
        == "high detail, best quality, blue eyes, looking at viewer"
    )
