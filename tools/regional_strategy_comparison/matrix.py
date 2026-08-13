# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the immutable P10.2 managed comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tools.anima_attention_coupling_prompts import (
    PINNED_ADAPTER_B,
    PINNED_ADAPTER_A,
    PINNED_NIJI,
    PINNED_VNTG,
    RegionalLora,
)

Strategy = Literal["regional_conditioning", "attention_coupling"]
SpatialProfile = Literal[
    "full",
    "tiled_multidiffusion",
    "tiled_mixture_of_diffusers",
    "contextual_multidiffusion",
    "contextual_mixture_of_diffusers",
]

SOURCE_CASE_ID = "full-attention-zero-lora"
MASK_CASE_ID = "vertical-hard-50-50"
SOURCE_WIDTH = 1024
SOURCE_HEIGHT = 1024
TARGET_WIDTH = 1536
TARGET_HEIGHT = 1536
STEPS = 12
CFG = 1.0
SEED = 1_029_384_756
SAMPLER = "er_sde"
SCHEDULER = "simple"
REFINEMENT_DENOISE = 0.3
SPATIAL_SIZE = 96
SPATIAL_OVERLAP = 32
SPATIAL_BATCH_SIZE = 4
CONTEXTUAL_GLOBAL_STEPS = 3


@dataclass(frozen=True, slots=True)
class StrategyComparisonCase:
    """Describe one public-node performance comparison case."""

    case_id: str
    label: str
    strategy: Strategy
    spatial_profile: SpatialProfile
    regional_loras: tuple[RegionalLora, ...] = ()

    @property
    def regional_lora_count(self) -> int:
        """Return the declared regional adapter count."""

        return len(self.regional_loras)

    @property
    def is_refinement(self) -> bool:
        """Return whether the case consumes the shared 1024 source."""

        return self.spatial_profile != "full"

    @property
    def diffusion_mode(self) -> str | None:
        """Return the selected tiled fusion mode when applicable."""

        if self.spatial_profile.endswith("multidiffusion"):
            return "multidiffusion"
        if self.spatial_profile.endswith("mixture_of_diffusers"):
            return "mixture_of_diffusers"
        return None


def cases() -> tuple[StrategyComparisonCase, ...]:
    """Return the closed 16-case P10.2 matrix in execution order."""

    one_lora = (RegionalLora(0, PINNED_ADAPTER_A, 0.8),)
    four_loras = (
        RegionalLora(0, PINNED_ADAPTER_A, 0.8),
        RegionalLora(0, PINNED_NIJI, 0.55),
        RegionalLora(1, PINNED_ADAPTER_B, 0.8),
        RegionalLora(1, PINNED_VNTG, 0.55),
    )
    definitions: list[StrategyComparisonCase] = []
    profiles: tuple[SpatialProfile, ...] = (
        "full",
        "tiled_multidiffusion",
        "tiled_mixture_of_diffusers",
        "contextual_multidiffusion",
        "contextual_mixture_of_diffusers",
    )
    for profile in profiles:
        readable = profile.replace("_", " ").title()
        definitions.extend(
            (
                StrategyComparisonCase(
                    case_id=f"{profile.replace('_', '-')}-attention-zero-lora",
                    label=f"{readable} — Attention Coupling — zero regional LoRAs",
                    strategy="attention_coupling",
                    spatial_profile=profile,
                ),
                StrategyComparisonCase(
                    case_id=f"{profile.replace('_', '-')}-regional-zero-lora",
                    label=f"{readable} — Regional Conditioning — zero regional LoRAs",
                    strategy="regional_conditioning",
                    spatial_profile=profile,
                ),
            )
        )
        if profile in {"full", "tiled_multidiffusion", "contextual_multidiffusion"}:
            definitions.extend(
                (
                    StrategyComparisonCase(
                        case_id=f"{profile.replace('_', '-')}-attention-one-lora",
                        label=f"{readable} — Attention Coupling — one ADAPTER_A LoRA",
                        strategy="attention_coupling",
                        spatial_profile=profile,
                        regional_loras=one_lora,
                    ),
                    StrategyComparisonCase(
                        case_id=f"{profile.replace('_', '-')}-attention-four-lora",
                        label=f"{readable} — Attention Coupling — four regional LoRAs",
                        strategy="attention_coupling",
                        spatial_profile=profile,
                        regional_loras=four_loras,
                    ),
                )
            )
    return tuple(definitions)


def expected_low_rank_multiplier(case: StrategyComparisonCase) -> float:
    """Return the exact Prompt Control execution-use multiplier."""

    return {0: 0.0, 1: 2.0, 4: 8.0}[case.regional_lora_count]
