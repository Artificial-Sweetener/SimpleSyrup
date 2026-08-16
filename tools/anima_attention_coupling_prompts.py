# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own pinned prompts and LoRA declarations for managed Anima workflows."""

from __future__ import annotations

from dataclasses import dataclass

PINNED_PRIMARY_ADAPTER = "Anima\\style\\adapter-a.safetensors"
PINNED_SECONDARY_ADAPTER = "Anima\\style\\adapter-b.safetensors"
PINNED_TERTIARY_ADAPTER = "Anima\\style\\adapter-c.safetensors"
PINNED_QUATERNARY_ADAPTER = "Anima\\style\\adapter-d.safetensors"
GLOBAL_GLOBAL_ADAPTER = "Anima\\style\\GLOBAL_ADAPTER_anima.safetensors"
NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, "
    "blurry, jpeg artifacts, bad anatomy, bad hands, extra digits, watermark"
)
GLOBAL_PROMPT = (
    "masterpiece, best quality, score_7, safe, two people standing side by "
    "side in a city street, full body, balanced composition"
)
REGIONAL_PROMPTS = (
    "1girl on the left, short black hair, red jacket, black skirt",
    "1boy on the right, short white hair, blue jacket, white trousers",
)


@dataclass(frozen=True, slots=True)
class GlobalLora:
    """Describe one conventional full-canvas model LoRA."""

    lora_name: str
    strength: float


@dataclass(frozen=True, slots=True)
class RegionalLora:
    """Describe one region-owned LoRA and its optional sampling interval."""

    region_index: int
    lora_name: str
    strength: float
    schedule: tuple[float, float] | None = None


def render_regional_positive_prompt(adapters: tuple[RegionalLora, ...]) -> str:
    """Render the exact global and region-segment Prompt Control prompt."""

    segments = [GLOBAL_PROMPT]
    for region_index, regional_prompt in enumerate(REGIONAL_PROMPTS):
        tags = " ".join(
            render_regional_lora_tag(adapter)
            for adapter in adapters
            if adapter.region_index == region_index
        )
        segments.append(f"{regional_prompt} {tags}".rstrip())
    return "[SEP]".join(segments)


def render_regional_lora_tag(adapter: RegionalLora) -> str:
    """Render one pinned Prompt Control static or scheduled LoRA tag."""

    tag = f"<lora:{adapter.lora_name}:{adapter.strength:g}>"
    if adapter.schedule is None:
        return tag
    start, end = adapter.schedule
    return f"[{tag}:{start:.2f},{end:.2f}]"
