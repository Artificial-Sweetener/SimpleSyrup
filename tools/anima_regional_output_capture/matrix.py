# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the fixed labeled Anima regional LoRA output matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from tools.attention_coupling_benchmark.comfy_probe.anima_regional_profile_spec import (
    VisualRegionalAdapter,
)

PINNED_ADAPTER_A = "Anima\\style\\adapter-a.safetensors"
PINNED_ADAPTER_B = "Anima\\style\\adapter-b.safetensors"
GLOBAL_GLOBAL_ADAPTER = "Anima\\style\\global-adapter.safetensors"
SEED = 1_029_384_756
WIDTH = 1024
HEIGHT = 1024
STEPS = 30
CFG = 1.0
SAMPLER = "er_sde"
SCHEDULER = "simple"
NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, "
    "blurry, jpeg artifacts, chromatic aberration, bad anatomy, bad hands, "
    "extra digits, fewer digits, watermark"
)
FIDELITY_PROMPT = (
    "masterpiece, best quality, score_7, safe, two people standing side by "
    "side in a city street, one with short black hair and a red jacket, one "
    "with short white hair and a blue jacket, full body"
)
SPLIT_GLOBAL_PROMPT = (
    "masterpiece, best quality, score_7, safe, two people standing side by "
    "side in a city street, full body, balanced composition"
)
SPLIT_REGIONAL_PROMPTS = (
    "1girl on the left, short black hair, red jacket, black skirt",
    "1boy on the right, short white hair, blue jacket, white trousers",
)


class VisualProfileKind(StrEnum):
    """Select ordinary global or optimized regional execution."""

    GLOBAL_REFERENCE = "global_reference"
    REGIONAL_OPTIMIZED = "regional_optimized"


@dataclass(frozen=True, slots=True)
class GlobalVisualAdapter:
    """Describe one conventional full-canvas model LoRA."""

    lora_name: str
    strength: float


@dataclass(frozen=True, slots=True)
class VisualOutputProfile:
    """Describe one fully labeled decoded-output workflow."""

    profile_id: str
    label: str
    kind: VisualProfileKind
    global_adapters: tuple[GlobalVisualAdapter, ...]
    regional_adapters: tuple[VisualRegionalAdapter, ...]
    global_prompt: str
    regional_prompts: tuple[str, ...]
    mask_case_id: str | None


def profiles() -> tuple[VisualOutputProfile, ...]:
    """Return fidelity controls plus a true global/left/right LoRA composition."""

    return (
        VisualOutputProfile(
            "global-reference-adapter_a",
            "Ordinary global ADAPTER_A reference",
            VisualProfileKind.GLOBAL_REFERENCE,
            (GlobalVisualAdapter(PINNED_ADAPTER_A, 1.0),),
            (),
            FIDELITY_PROMPT,
            (),
            None,
        ),
        VisualOutputProfile(
            "regional-optimized-adapter_a-all-one",
            "Optimized regional ADAPTER_A with an all-one mask",
            VisualProfileKind.REGIONAL_OPTIMIZED,
            (),
            (VisualRegionalAdapter(0, RegionalLoraBranch.POSITIVE, PINNED_ADAPTER_A, 1.0),),
            FIDELITY_PROMPT,
            (FIDELITY_PROMPT,),
            "all-one",
        ),
        VisualOutputProfile(
            "regional-optimized-adapter_a-four-all-one",
            "Optimized four-adapter ADAPTER_A composition with an all-one mask",
            VisualProfileKind.REGIONAL_OPTIMIZED,
            (),
            tuple(
                VisualRegionalAdapter(0, RegionalLoraBranch.POSITIVE, PINNED_ADAPTER_A, 0.25)
                for _index in range(4)
            ),
            FIDELITY_PROMPT,
            (FIDELITY_PROMPT,),
            "all-one",
        ),
        VisualOutputProfile(
            "regional-split-global-global_adapter-left-adapter_a-right-adapter_b",
            "Global GLOBAL_ADAPTER with left ADAPTER_A and right ADAPTER_B",
            VisualProfileKind.REGIONAL_OPTIMIZED,
            (GlobalVisualAdapter(GLOBAL_GLOBAL_ADAPTER, 0.35),),
            (
                VisualRegionalAdapter(0, RegionalLoraBranch.POSITIVE, PINNED_ADAPTER_A, 0.8),
                VisualRegionalAdapter(1, RegionalLoraBranch.POSITIVE, PINNED_ADAPTER_B, 0.8),
            ),
            SPLIT_GLOBAL_PROMPT,
            SPLIT_REGIONAL_PROMPTS,
            "vertical-hard-50-50",
        ),
    )
