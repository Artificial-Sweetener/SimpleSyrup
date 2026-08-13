# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bind immutable P0.8 cases to fixed P9.1 sampling fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from tools.prompt_control_characterization.cases import (
    PromptControlCase,
)
from tools.prompt_control_characterization.cases import (
    cases as characterization_cases,
)

PUBLIC_NODE_ID = "SimpleSyrup.KSamplerAttentionCoupling"
WIDTH = 512
HEIGHT = 512
STEPS = 8
CFG = 4.0
MASK_CASE_ID = "vertical-hard-50-50"
BASE_PROMPT = (
    "masterpiece, best quality, score_7, safe, two distinct people standing "
    "side by side, full body, balanced city-street composition"
)
REGION_ONE_PROMPT = "one calm person on the right wearing a blue jacket"
NEGATIVE_PROMPT = "low quality, blurry, bad anatomy, fused people, duplicate subject"
REGION_ZERO_NEGATIVE_PROMPT = "low quality"


@dataclass(frozen=True, slots=True)
class PromptControlAttentionCase:
    """Add only P9.1 fixture values to one authoritative P0.8 case."""

    characterization: PromptControlCase
    label: str
    cfg: float = CFG
    feather: int = 0

    @property
    def case_id(self) -> str:
        """Return the unchanged P0.8 case identity."""

        return self.characterization.case_id

    @property
    def adapter_identities(self) -> tuple[str, ...]:
        """Return stable probe labels for regional model hooks."""

        if self.characterization.expect_static_model_lora:
            return ("adapter_a-static-0.75",)
        return tuple(
            adapter.identity for adapter in self.characterization.expected_adapters
        )

    @property
    def has_regional_hooks(self) -> bool:
        """Return whether Prompt Control emits a supported regional HookGroup."""

        return bool(self.adapter_identities)


def cases() -> tuple[PromptControlAttentionCase, ...]:
    """Return all nine P0.8 cases in their immutable authoritative order."""

    return tuple(
        PromptControlAttentionCase(
            characterization=case,
            label=(
                f"Prompt Control {case.case_id} through full Anima Attention Coupling"
            ),
        )
        for case in characterization_cases()
    )
