# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the fixed global/regional PRIMARY_ADAPTER visual comparison matrix."""

from __future__ import annotations

import re
from dataclasses import dataclass

PRIMARY_ADAPTER_NAME = r"Anima\style\adapter-a.safetensors"
_PRIMARY_ADAPTER_TAG = re.compile(rf"\s*<lora:{re.escape(PRIMARY_ADAPTER_NAME)}:[^>]+>")


@dataclass(frozen=True, slots=True)
class GlobalLoraVisualCase:
    """Describe one fixed-seed PRIMARY_ADAPTER placement and expected outcome."""

    case_id: str
    label: str
    global_strength: float | None = None
    regional_strength: float | None = None
    expect_overlap_rejection: bool = False


def cases() -> tuple[GlobalLoraVisualCase, ...]:
    """Return baseline, global strengths, regional, and duplicate coverage."""

    return (
        GlobalLoraVisualCase("no-lora", "NO LORA — reference checkpoint baseline"),
        GlobalLoraVisualCase(
            "global-primary_adapter-050",
            "GLOBAL PRIMARY_ADAPTER 0.5 — full image",
            global_strength=0.5,
        ),
        GlobalLoraVisualCase(
            "global-primary_adapter-100",
            "GLOBAL PRIMARY_ADAPTER 1.0 — full image",
            global_strength=1.0,
        ),
        GlobalLoraVisualCase(
            "regional-primary_adapter-100-left",
            "REGIONAL PRIMARY_ADAPTER 1.0 — left only",
            regional_strength=1.0,
        ),
        GlobalLoraVisualCase(
            "duplicate-global-regional-primary_adapter-100",
            "GLOBAL + LEFT PRIMARY_ADAPTER 1.0 — duplicate",
            global_strength=1.0,
            regional_strength=1.0,
            expect_overlap_rejection=True,
        ),
    )


def render_positive_prompt(
    source_prompt: str,
    case: GlobalLoraVisualCase,
) -> str:
    """Place the adapter in only the segments declared by a case."""

    segments = [
        _PRIMARY_ADAPTER_TAG.sub("", segment).strip()
        for segment in source_prompt.split("[SEP]")
    ]
    if len(segments) != 3:
        raise ValueError("Global LoRA visual proof requires three prompt segments.")
    if case.regional_strength is not None:
        segments[1] = f"{segments[1]} {_tag(case.regional_strength)}"
    return "[SEP]".join(segments)


def _tag(strength: float) -> str:
    """Render one exact Prompt Control PRIMARY_ADAPTER tag."""

    return f"<lora:{PRIMARY_ADAPTER_NAME}:{strength:g}>"
