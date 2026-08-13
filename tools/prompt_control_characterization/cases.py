# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable Prompt Control conditioning and LoRA schedule cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TextConstruction = Literal["lazy", "adjacent", "overlapping", "inactive"]

ADAPTER_A = "Anima\\style\\adapter-a.safetensors"
ADAPTER_B = "Anima\\style\\adapter-b.safetensors"


@dataclass(frozen=True)
class ExpectedConditioning:
    """Describe one expected conditioning interval and optional strength."""

    start: float
    end: float
    strength: float | None = None


@dataclass(frozen=True)
class ExpectedAdapter:
    """Describe one expected ordered adapter and native keyframe schedule."""

    identity: str
    strength_model: float
    strength_clip: float
    keyframes: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PromptControlCase:
    """Define one host-facing characterization execution."""

    case_id: str
    text_construction: TextConstruction
    positive_text: str
    lora_text: str
    expected_positive: tuple[ExpectedConditioning, ...]
    expected_adapters: tuple[ExpectedAdapter, ...] = ()
    expect_static_model_lora: bool = False
    expected_negative: tuple[ExpectedConditioning, ...] = (
        ExpectedConditioning(0.0, 1.0),
    )
    expected_model_call_count: int = 8
    expected_runtime_transitions: tuple[tuple[int, tuple[float, ...]], ...] = ((1, ()),)


def cases() -> tuple[PromptControlCase, ...]:
    """Return the complete static, boundary, inactive, and LoRA matrix."""

    return (
        PromptControlCase(
            "text-static",
            "lazy",
            "a calm subject:0.6",
            "",
            (ExpectedConditioning(0.0, 1.0, 0.6),),
        ),
        PromptControlCase(
            "text-adjacent",
            "adjacent",
            "",
            "",
            (ExpectedConditioning(0.0, 0.5), ExpectedConditioning(0.5, 1.0)),
        ),
        PromptControlCase(
            "text-overlapping",
            "overlapping",
            "",
            "",
            (ExpectedConditioning(0.0, 0.75), ExpectedConditioning(0.25, 1.0)),
        ),
        PromptControlCase(
            "text-inactive",
            "inactive",
            "inactive subject",
            "",
            (ExpectedConditioning(0.5, 0.5),),
        ),
        PromptControlCase(
            "lora-single-static",
            "lazy",
            "a calm subject",
            f"<lora:{ADAPTER_A}:0.75:0.25>",
            (ExpectedConditioning(0.0, 1.0),),
            expect_static_model_lora=True,
        ),
        PromptControlCase(
            "lora-single-scheduled",
            "lazy",
            "a calm subject",
            f"[<lora:{ADAPTER_A}:0.75:0.25>:0.25,0.75]",
            (ExpectedConditioning(0.0, 1.0),),
            (
                ExpectedAdapter(
                    "adapter_a-0.75-0.25",
                    0.75,
                    0.25,
                    ((0.0, 0.0), (0.25, 1.0), (0.75, 0.0)),
                ),
            ),
            expected_negative=(ExpectedConditioning(0.75, 1.0),),
            expected_model_call_count=10,
            expected_runtime_transitions=((1, (0.0,)), (3, (0.75,)), (7, (0.0,))),
        ),
        PromptControlCase(
            "lora-adjacent",
            "lazy",
            "a calm subject",
            f"[<lora:{ADAPTER_A}:0.4>:0.0,0.5] [<lora:{ADAPTER_B}:0.6>:0.5,1.0]",
            (ExpectedConditioning(0.0, 1.0),),
            (
                ExpectedAdapter("adapter_a-0.4", 0.4, 0.4, ((0.0, 1.0), (0.5, 0.0))),
                ExpectedAdapter("adapter_b-0.6", 0.6, 0.6, ((0.0, 0.0), (0.5, 1.0))),
            ),
            expected_negative=(ExpectedConditioning(0.5, 1.0),),
            expected_model_call_count=12,
            expected_runtime_transitions=((1, (0.4, 0.0)), (5, (0.0, 0.6))),
        ),
        PromptControlCase(
            "lora-stacked-overlap",
            "lazy",
            "a calm subject",
            f"[<lora:{ADAPTER_A}:0.4>:0.0,0.75] [<lora:{ADAPTER_B}:0.6>:0.25,1.0]",
            (ExpectedConditioning(0.0, 1.0),),
            (
                ExpectedAdapter(
                    "adapter_a-0.4",
                    0.4,
                    0.4,
                    ((0.0, 1.0), (0.25, 0.0), (0.25, 1.0), (0.75, 0.0)),
                ),
                ExpectedAdapter(
                    "adapter_b-0.6",
                    0.6,
                    0.6,
                    ((0.0, 0.0), (0.25, 1.0), (0.75, 0.0), (0.75, 1.0)),
                ),
            ),
            expected_negative=(ExpectedConditioning(0.75, 1.0),),
            expected_model_call_count=10,
            expected_runtime_transitions=(
                (1, (0.4, 0.0)),
                (3, (0.0, 0.6)),
                (4, (0.4, 0.6)),
                (7, (0.0, 0.0)),
                (9, (0.0, 0.6)),
            ),
        ),
        PromptControlCase(
            "lora-inactive",
            "lazy",
            "a calm subject",
            f"[<lora:{ADAPTER_A}:0.5>:0.5,0.5]",
            (ExpectedConditioning(0.0, 1.0),),
        ),
    )
