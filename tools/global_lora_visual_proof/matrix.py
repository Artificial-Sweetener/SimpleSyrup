"""Define the fixed global/regional ADAPTER_A visual comparison matrix."""

from __future__ import annotations

import re
from dataclasses import dataclass

ADAPTER_A_NAME = r"Anima\style\adapter-a.safetensors"
_ADAPTER_A_TAG = re.compile(r"\s*<lora:Anima\\style\\ADAPTER_A_anima\.safetensors:[^>]+>")


@dataclass(frozen=True, slots=True)
class GlobalLoraVisualCase:
    """Describe one fixed-seed ADAPTER_A placement and expected outcome."""

    case_id: str
    label: str
    global_strength: float | None = None
    regional_strength: float | None = None
    expect_overlap_rejection: bool = False


def cases() -> tuple[GlobalLoraVisualCase, ...]:
    """Return baseline, global strengths, regional, and duplicate coverage."""

    return (
        GlobalLoraVisualCase("no-lora", "NO LORA — CHECKPOINT_A baseline"),
        GlobalLoraVisualCase(
            "global-adapter_a-050",
            "GLOBAL ADAPTER_A 0.5 — full image",
            global_strength=0.5,
        ),
        GlobalLoraVisualCase(
            "global-adapter_a-100",
            "GLOBAL ADAPTER_A 1.0 — full image",
            global_strength=1.0,
        ),
        GlobalLoraVisualCase(
            "regional-adapter_a-100-left",
            "REGIONAL ADAPTER_A 1.0 — left only",
            regional_strength=1.0,
        ),
        GlobalLoraVisualCase(
            "duplicate-global-regional-adapter_a-100",
            "GLOBAL + LEFT ADAPTER_A 1.0 — duplicate",
            global_strength=1.0,
            regional_strength=1.0,
            expect_overlap_rejection=True,
        ),
    )


def render_positive_prompt(
    source_prompt: str,
    case: GlobalLoraVisualCase,
) -> str:
    """Place ADAPTER_A only in the global and regional segments declared by a case."""

    segments = [
        _ADAPTER_A_TAG.sub("", segment).strip() for segment in source_prompt.split("[SEP]")
    ]
    if len(segments) != 3:
        raise ValueError("Global LoRA visual proof requires three prompt segments.")
    if case.regional_strength is not None:
        segments[1] = f"{segments[1]} {_tag(case.regional_strength)}"
    return "[SEP]".join(segments)


def _tag(strength: float) -> str:
    """Render one exact Prompt Control ADAPTER_A tag."""

    return f"<lora:{ADAPTER_A_NAME}:{strength:g}>"
