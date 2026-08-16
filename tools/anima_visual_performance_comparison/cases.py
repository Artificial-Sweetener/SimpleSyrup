# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare the exact three-case Anima visual performance comparison."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AnimaVisualMode(StrEnum):
    """Select native, regional-attention, or regional-LoRA execution."""

    PLAIN = "plain-native"
    REGIONAL_ATTENTION = "regional-attention"
    REGIONAL_LORA = "regional-lora-left"


@dataclass(frozen=True, slots=True)
class AnimaVisualCase:
    """Name one visibly labeled execution topology."""

    mode: AnimaVisualMode
    label: str


def visual_cases() -> tuple[AnimaVisualCase, ...]:
    """Return the predeclared native, attention, and one-LoRA order."""

    return (
        AnimaVisualCase(
            AnimaVisualMode.PLAIN,
            "A — Plain native Anima — no regions, no LoRA",
        ),
        AnimaVisualCase(
            AnimaVisualMode.REGIONAL_ATTENTION,
            "B — Anima regional attention — no LoRA",
        ),
        AnimaVisualCase(
            AnimaVisualMode.REGIONAL_LORA,
            "C — Anima regional attention — left LoRA 1.0",
        ),
    )
