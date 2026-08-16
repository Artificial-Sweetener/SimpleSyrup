# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the immutable P5.9 public Attention Coupling workflow matrix."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_prompts import (
    GLOBAL_GLOBAL_ADAPTER,
    PINNED_PRIMARY_ADAPTER,
    PINNED_QUATERNARY_ADAPTER,
    PINNED_SECONDARY_ADAPTER,
    PINNED_TERTIARY_ADAPTER,
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

    primary_adapter_left = RegionalLora(0, PINNED_PRIMARY_ADAPTER, 0.8)
    secondary_adapter_right = RegionalLora(1, PINNED_SECONDARY_ADAPTER, 0.8)
    return (
        IntegrationCase(
            "zero-regional-hard-cfg1",
            "Zero regional LoRAs; hard left/right Attention Coupling; CFG 1",
            1.0,
            "vertical-hard-50-50",
            0,
        ),
        IntegrationCase(
            "one-regional-primary_adapter-left-hard-cfg1",
            "Left regional LoRA; right attention only; hard masks; CFG 1",
            1.0,
            "vertical-hard-50-50",
            0,
            regional_loras=(primary_adapter_left,),
        ),
        IntegrationCase(
            "global-global_adapter-left-primary_adapter-right-secondary_adapter-hard-cfg1",
            "Global plus distinct left/right regional LoRAs; CFG 1",
            1.0,
            "vertical-hard-50-50",
            0,
            global_loras=(GlobalLora(GLOBAL_GLOBAL_ADAPTER, 0.35),),
            regional_loras=(primary_adapter_left, secondary_adapter_right),
        ),
        IntegrationCase(
            "four-regional-stacked-hard-cfg1",
            "Four regional LoRAs: two left and two right; CFG 1",
            1.0,
            "vertical-hard-50-50",
            0,
            regional_loras=(
                RegionalLora(0, PINNED_PRIMARY_ADAPTER, 0.55),
                RegionalLora(0, PINNED_TERTIARY_ADAPTER, 0.45),
                RegionalLora(1, PINNED_SECONDARY_ADAPTER, 0.55),
                RegionalLora(1, PINNED_QUATERNARY_ADAPTER, 0.45),
            ),
        ),
        IntegrationCase(
            "adjacent-schedules-hard-cfg1",
            "Left LoRA steps 0-50%, then right LoRA steps 50-100%; hard masks",
            1.0,
            "vertical-hard-50-50",
            0,
            regional_loras=(
                RegionalLora(0, PINNED_PRIMARY_ADAPTER, 0.8, (0.0, 0.5)),
                RegionalLora(1, PINNED_SECONDARY_ADAPTER, 0.8, (0.5, 1.0)),
            ),
        ),
        IntegrationCase(
            "overlapping-schedules-and-masks-cfg1",
            "Overlapping LoRA schedules and spatial masks; CFG 1",
            1.0,
            "overlapping-regions",
            0,
            regional_loras=(
                RegionalLora(0, PINNED_PRIMARY_ADAPTER, 0.8, (0.0, 0.75)),
                RegionalLora(1, PINNED_SECONDARY_ADAPTER, 0.8, (0.25, 1.0)),
            ),
        ),
        IntegrationCase(
            "two-regional-feathered-cfg1",
            "Distinct left/right LoRAs with 96-pixel feathering; CFG 1",
            1.0,
            "vertical-hard-50-50",
            96,
            regional_loras=(primary_adapter_left, secondary_adapter_right),
        ),
        IntegrationCase(
            "two-regional-hard-cfg4",
            "Distinct left/right LoRAs with ordinary positive/negative CFG 4",
            4.0,
            "vertical-hard-50-50",
            0,
            regional_loras=(primary_adapter_left, secondary_adapter_right),
        ),
    )
