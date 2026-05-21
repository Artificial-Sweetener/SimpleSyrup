# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Prompt Encode Style ComfyUI node declarations."""

from __future__ import annotations

from typing import Any

import pytest

from simple_syrup.nodes.prompt_encode_style import PromptEncodeStyle
from simple_syrup.nodes.prompt_encode_style_and_normalization import (
    PromptEncodeStyleAndNormalization,
)

ENCODE_STYLE_OPTIONS = [
    "A1111",
    "Comfy",
    "Comfy++",
    "Compel",
    "Down Weight",
    "Perp",
]


def test_prompt_encode_style_node_contract_constants() -> None:
    """Style-only node constants match the public ComfyUI contract."""

    assert PromptEncodeStyle.RETURN_TYPES == ("STRING",)
    assert PromptEncodeStyle.RETURN_NAMES == ("style_tag",)
    assert PromptEncodeStyle.FUNCTION == "build"
    assert PromptEncodeStyle.CATEGORY == "SimpleSyrup/Prompt"


def test_prompt_encode_style_node_declares_expected_inputs() -> None:
    """Style-only node input declaration includes only encode style."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = PromptEncodeStyle.INPUT_TYPES()

    required = input_types["required"]

    assert set(required) == {"encode_style"}
    assert required["encode_style"][0] == ENCODE_STYLE_OPTIONS
    assert required["encode_style"][1]["default"] == "A1111"
    assert "tooltip" in required["encode_style"][1]


@pytest.mark.parametrize(
    ("encode_style", "expected"),
    [
        ("A1111", "STYLE(A1111) "),
        ("Comfy++", "STYLE(comfy++) "),
        ("Down Weight", "STYLE(down_weight) "),
    ],
)
def test_prompt_encode_style_node_builds_style_tag(
    encode_style: str, expected: str
) -> None:
    """Style-only node formats Prompt Control STYLE tags from combo values."""

    (style_tag,) = PromptEncodeStyle().build(encode_style)

    assert style_tag == expected
    assert style_tag.endswith(" ")
    assert not style_tag.endswith("  ")


def test_prompt_encode_style_and_normalization_node_contract_constants() -> None:
    """Normalization node constants match the public ComfyUI contract."""

    assert PromptEncodeStyleAndNormalization.RETURN_TYPES == ("STRING",)
    assert PromptEncodeStyleAndNormalization.RETURN_NAMES == ("style_tag",)
    assert PromptEncodeStyleAndNormalization.FUNCTION == "build"
    assert PromptEncodeStyleAndNormalization.CATEGORY == "SimpleSyrup/Prompt"


def test_prompt_encode_style_and_normalization_node_declares_expected_inputs() -> None:
    """Normalization node input declaration includes encode style and normalization."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = (
        PromptEncodeStyleAndNormalization.INPUT_TYPES()
    )

    required = input_types["required"]

    assert set(required) == {"encode_style", "normalization"}
    assert required["encode_style"][0] == ENCODE_STYLE_OPTIONS
    assert required["encode_style"][1]["default"] == "A1111"
    assert "tooltip" in required["encode_style"][1]
    assert required["normalization"][0] == ["none", "length", "mean", "length+mean"]
    assert required["normalization"][1]["default"] == "none"
    assert "tooltip" in required["normalization"][1]


@pytest.mark.parametrize(
    ("encode_style", "normalization", "expected"),
    [
        ("A1111", "none", "STYLE(A1111) "),
        ("A1111", "length", "STYLE(A1111, length) "),
        ("Comfy++", "none", "STYLE(comfy++) "),
        ("Comfy++", "length", "STYLE(comfy++, length) "),
        ("Down Weight", "none", "STYLE(down_weight) "),
        ("Perp", "length+mean", "STYLE(perp, length+mean) "),
    ],
)
def test_prompt_encode_style_and_normalization_node_builds_style_tag(
    encode_style: str, normalization: str, expected: str
) -> None:
    """Normalization node formats Prompt Control STYLE tags from combo values."""

    (style_tag,) = PromptEncodeStyleAndNormalization().build(
        encode_style, normalization
    )

    assert style_tag == expected
    assert style_tag.endswith(" ")
    assert not style_tag.endswith("  ")
