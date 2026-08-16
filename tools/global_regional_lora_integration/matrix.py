# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the closed P9.3 global/regional Anima LoRA matrix."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_integration.matrix import IntegrationCase
from tools.anima_attention_coupling_prompts import (
    GLOBAL_GLOBAL_ADAPTER,
    PINNED_PRIMARY_ADAPTER,
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

    global_primary_adapter = GlobalLora(PINNED_PRIMARY_ADAPTER, 0.35)
    regional_primary_adapter = RegionalLora(0, PINNED_PRIMARY_ADAPTER, 0.8)
    return (
        GlobalRegionalLoraCase(
            IntegrationCase(
                "global-global_adapter-regional-primary_adapter-before",
                "Distinct global and regional adapters before transition",
                1.0,
                MASK_CASE_ID,
                0,
                global_loras=(GlobalLora(GLOBAL_GLOBAL_ADAPTER, 0.35),),
                regional_loras=(regional_primary_adapter,),
            )
        ),
        GlobalRegionalLoraCase(
            IntegrationCase(
                "regional-primary_adapter-only",
                "Regional PRIMARY_ADAPTER only through Anima Attention Coupling",
                1.0,
                MASK_CASE_ID,
                0,
                regional_loras=(regional_primary_adapter,),
            )
        ),
        GlobalRegionalLoraCase(
            IntegrationCase(
                "global-global_adapter-regional-primary_adapter-after",
                "Distinct global and regional adapters after transition",
                1.0,
                MASK_CASE_ID,
                0,
                global_loras=(GlobalLora(GLOBAL_GLOBAL_ADAPTER, 0.35),),
                regional_loras=(regional_primary_adapter,),
            )
        ),
        GlobalRegionalLoraCase(
            IntegrationCase(
                "global-primary_adapter-only",
                "Global PRIMARY_ADAPTER only through Anima Attention Coupling",
                1.0,
                MASK_CASE_ID,
                0,
                global_loras=(global_primary_adapter,),
            )
        ),
        GlobalRegionalLoraCase(
            IntegrationCase(
                "duplicate-global-regional-primary_adapter",
                "Exact global and regional PRIMARY_ADAPTER duplicate rejection",
                1.0,
                MASK_CASE_ID,
                0,
                global_loras=(global_primary_adapter,),
                regional_loras=(regional_primary_adapter,),
            ),
            expect_overlap_error=True,
        ),
    )
