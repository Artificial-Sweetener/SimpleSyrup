# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the immutable P5.9 public Attention Coupling workflow matrix."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_prompts import (
    GLOBAL_GLOBAL_ADAPTER,
    PINNED_ADAPTER_B,
    PINNED_ADAPTER_A,
    PINNED_NIJI,
    PINNED_VNTG,
    GlobalLora,
    RegionalLora,
)

PUBLIC_NODE_ID = "SimpleSyrup.KSamplerAttentionCoupling"


@dataclass(frozen=True, slots=True)
class IntegrationCase:
    """Describe one complete public-node image and evidence case."""

    case_id: str
    label: str
    cfg: float
    mask_case_id: str
    feather: int
    global_loras: tuple[GlobalLora, ...] = ()
    regional_loras: tuple[RegionalLora, ...] = ()

    @property
    def regional_lora_count(self) -> int:
        """Return the declared model-side regional adapter count."""

        return len(self.regional_loras)


def cases() -> tuple[IntegrationCase, ...]:
    """Return complete count, schedule, mask, overlap, and CFG coverage."""

    adapter_a_left = RegionalLora(0, PINNED_ADAPTER_A, 0.8)
    adapter_b_right = RegionalLora(1, PINNED_ADAPTER_B, 0.8)
    return (
        IntegrationCase(
            "zero-regional-hard-cfg1",
            "Zero regional LoRAs; hard left/right Attention Coupling; CFG 1",
            1.0,
            "vertical-hard-50-50",
            0,
        ),
        IntegrationCase(
            "one-regional-adapter_a-left-hard-cfg1",
            "Left ADAPTER_A regional LoRA; right attention only; hard masks; CFG 1",
            1.0,
            "vertical-hard-50-50",
            0,
            regional_loras=(adapter_a_left,),
        ),
        IntegrationCase(
            "global-global_adapter-left-adapter_a-right-adapter_b-hard-cfg1",
            "Global GLOBAL_ADAPTER plus left ADAPTER_A and right ADAPTER_B regional LoRAs; CFG 1",
            1.0,
            "vertical-hard-50-50",
            0,
            global_loras=(GlobalLora(GLOBAL_GLOBAL_ADAPTER, 0.35),),
            regional_loras=(adapter_a_left, adapter_b_right),
        ),
        IntegrationCase(
            "four-regional-stacked-hard-cfg1",
            "Four regional LoRAs: ADAPTER_A and NIJI left, ADAPTER_B and VNTG right; CFG 1",
            1.0,
            "vertical-hard-50-50",
            0,
            regional_loras=(
                RegionalLora(0, PINNED_ADAPTER_A, 0.55),
                RegionalLora(0, PINNED_NIJI, 0.45),
                RegionalLora(1, PINNED_ADAPTER_B, 0.55),
                RegionalLora(1, PINNED_VNTG, 0.45),
            ),
        ),
        IntegrationCase(
            "adjacent-schedules-hard-cfg1",
            "Left ADAPTER_A steps 0-50%, then right ADAPTER_B steps 50-100%; hard masks",
            1.0,
            "vertical-hard-50-50",
            0,
            regional_loras=(
                RegionalLora(0, PINNED_ADAPTER_A, 0.8, (0.0, 0.5)),
                RegionalLora(1, PINNED_ADAPTER_B, 0.8, (0.5, 1.0)),
            ),
        ),
        IntegrationCase(
            "overlapping-schedules-and-masks-cfg1",
            "Overlapping ADAPTER_A/ADAPTER_B schedules and overlapping spatial masks; CFG 1",
            1.0,
            "overlapping-regions",
            0,
            regional_loras=(
                RegionalLora(0, PINNED_ADAPTER_A, 0.8, (0.0, 0.75)),
                RegionalLora(1, PINNED_ADAPTER_B, 0.8, (0.25, 1.0)),
            ),
        ),
        IntegrationCase(
            "two-regional-feathered-cfg1",
            "Left ADAPTER_A and right ADAPTER_B with 96-pixel feathering; CFG 1",
            1.0,
            "vertical-hard-50-50",
            96,
            regional_loras=(adapter_a_left, adapter_b_right),
        ),
        IntegrationCase(
            "two-regional-hard-cfg4",
            "Left ADAPTER_A and right ADAPTER_B with ordinary positive/negative CFG 4",
            4.0,
            "vertical-hard-50-50",
            0,
            regional_loras=(adapter_a_left, adapter_b_right),
        ),
    )
