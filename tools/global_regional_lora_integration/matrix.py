# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the closed P9.3 global/regional Anima LoRA matrix."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_integration.matrix import IntegrationCase
from tools.anima_attention_coupling_prompts import (
    GLOBAL_GLOBAL_ADAPTER,
    PINNED_ADAPTER_A,
    GlobalLora,
    RegionalLora,
)

MASK_CASE_ID = "vertical-hard-50-50"


@dataclass(frozen=True, slots=True)
class GlobalRegionalLoraCase:
    """Bind one public workflow case to its expected terminal status."""

    integration: IntegrationCase
    expect_overlap_error: bool = False

    @property
    def case_id(self) -> str:
        """Return the stable artifact identity."""

        return self.integration.case_id


def cases() -> tuple[GlobalRegionalLoraCase, ...]:
    """Return ordered clean, transition, global-only, and duplicate coverage."""

    global_adapter_a = GlobalLora(PINNED_ADAPTER_A, 0.35)
    regional_adapter_a = RegionalLora(0, PINNED_ADAPTER_A, 0.8)
    return (
        GlobalRegionalLoraCase(
            IntegrationCase(
                "global-global_adapter-regional-adapter_a-before",
                "Distinct global GLOBAL_ADAPTER and regional ADAPTER_A before transition",
                1.0,
                MASK_CASE_ID,
                0,
                global_loras=(GlobalLora(GLOBAL_GLOBAL_ADAPTER, 0.35),),
                regional_loras=(regional_adapter_a,),
            )
        ),
        GlobalRegionalLoraCase(
            IntegrationCase(
                "regional-adapter_a-only",
                "Regional ADAPTER_A only through Anima Attention Coupling",
                1.0,
                MASK_CASE_ID,
                0,
                regional_loras=(regional_adapter_a,),
            )
        ),
        GlobalRegionalLoraCase(
            IntegrationCase(
                "global-global_adapter-regional-adapter_a-after",
                "Distinct global GLOBAL_ADAPTER and regional ADAPTER_A after transition",
                1.0,
                MASK_CASE_ID,
                0,
                global_loras=(GlobalLora(GLOBAL_GLOBAL_ADAPTER, 0.35),),
                regional_loras=(regional_adapter_a,),
            )
        ),
        GlobalRegionalLoraCase(
            IntegrationCase(
                "global-adapter_a-only",
                "Global ADAPTER_A only through Anima Attention Coupling",
                1.0,
                MASK_CASE_ID,
                0,
                global_loras=(global_adapter_a,),
            )
        ),
        GlobalRegionalLoraCase(
            IntegrationCase(
                "duplicate-global-regional-adapter_a",
                "Exact global and regional ADAPTER_A duplicate rejection",
                1.0,
                MASK_CASE_ID,
                0,
                global_loras=(global_adapter_a,),
                regional_loras=(regional_adapter_a,),
            ),
            expect_overlap_error=True,
        ),
    )
