# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify automatic NegPiP trigger detection."""

from __future__ import annotations

import pytest

from simple_syrup.domain.negative_prompt_weights import (
    contains_negative_prompt_weight,
)


@pytest.mark.parametrize(
    "text",
    [
        "(1girl:-2.00)",
        "portrait, (red jacket: -1.5)",
        "((nested):-0.25)",
        "[plain:(scheduled concept:-1.0):0.5]",
        "outer ((inner):-3.0)",
    ],
)
def test_detector_admits_effective_negative_prompt_weights(text: str) -> None:
    """Recognize valid negative emphasis wherever Prompt Control may schedule it."""

    assert contains_negative_prompt_weight(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "1girl:-2.00",
        "(1girl:2.00)",
        "(1girl)",
        "(1girl:not-a-number)",
        r"escaped \(1girl:-2.0\)",
        "unfinished (1girl:-2.0",
        "STYLE(A1111, length)",
        "",
    ],
)
def test_detector_rejects_non_negative_weight_syntax(text: str) -> None:
    """Do not activate for plain text, positive weights, escapes, or malformed input."""

    assert contains_negative_prompt_weight(text) is False


def test_detector_resolves_nested_effective_weight() -> None:
    """An inner explicit positive weight overrides a negative outer emphasis."""

    assert contains_negative_prompt_weight("((kept positive:2.0):-3.0)") is False


def test_detector_requires_text() -> None:
    """Reject dynamic non-text values before prompt planning."""

    with pytest.raises(TypeError, match="requires text"):
        contains_negative_prompt_weight(object())  # type: ignore[arg-type]
