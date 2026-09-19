# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detect effective negative weights in Comfy-style prompt emphasis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _WeightedPromptSegment:
    """Retain one parsed prompt fragment and its effective scalar weight."""

    text: str
    weight: float


def contains_negative_prompt_weight(text: str) -> bool:
    """Return whether valid nested emphasis gives any prompt text a negative weight."""

    if not isinstance(text, str):
        raise TypeError("Negative prompt-weight detection requires text.")
    escaped = text.replace(r"\)", "\0\1").replace(r"\(", "\0\2")
    return any(
        segment.text and segment.weight < 0.0
        for segment in _weighted_segments(escaped, 1.0)
    )


def _weighted_segments(
    text: str,
    current_weight: float,
) -> tuple[_WeightedPromptSegment, ...]:
    """Parse emphasis with the same nesting and final-colon rules as ComfyUI."""

    parsed: list[_WeightedPromptSegment] = []
    for item in _parenthesized_items(text):
        weight = current_weight
        if len(item) >= 2 and item[0] == "(" and item[-1] == ")":
            inner = item[1:-1]
            delimiter = inner.rfind(":")
            weight *= 1.1
            if delimiter > 0:
                try:
                    weight = float(inner[delimiter + 1 :])
                except ValueError:
                    pass
                else:
                    inner = inner[:delimiter]
            parsed.extend(_weighted_segments(inner, weight))
            continue
        parsed.append(
            _WeightedPromptSegment(
                item.replace("\0\1", ")").replace("\0\2", "("),
                current_weight,
            )
        )
    return tuple(parsed)


def _parenthesized_items(text: str) -> tuple[str, ...]:
    """Split top-level parenthesized regions while preserving malformed input."""

    result: list[str] = []
    current = ""
    nesting = 0
    for character in text:
        if character == "(":
            if nesting == 0:
                if current:
                    result.append(current)
                current = "("
            else:
                current += character
            nesting += 1
        elif character == ")":
            nesting -= 1
            if nesting == 0:
                result.append(f"{current})")
                current = ""
            else:
                current += character
        else:
            current += character
    if current:
        result.append(current)
    return tuple(result)
