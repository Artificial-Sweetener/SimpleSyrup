# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable model-neutral declarations for SDXL visual cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .matrix import REGIONAL_PROMPT_WEIGHT
from .visual_prompt_defaults import (
    BASE_NEGATIVE_G,
    BASE_NEGATIVE_L,
    BASE_POSITIVE_G,
    BASE_POSITIVE_L,
    LEFT_BASE_G,
    LEFT_BASE_L,
    RIGHT_BASE_G,
    RIGHT_BASE_L,
)


class VisualMaskProfile(StrEnum):
    """Name one exact mask geometry owned by the managed mask writer."""

    HARD = "hard"
    SOFT_OVERLAP = "soft-overlap"
    UNCOVERED_CENTER = "uncovered-center"


class VisualMode(StrEnum):
    """Name one public sampler geometry required by a visual case."""

    FULL = "full"
    TILED = "tiled-1.5x"
    CONTEXTUAL = "contextual-1.5x"


@dataclass(frozen=True, slots=True)
class RegionalVisualAdapter:
    """Declare one ordered regional adapter use and its schedule."""

    lora_name: str
    model_strength: float
    clip_strength: float
    schedule: tuple[tuple[float, float], ...] = ((0.0, 1.0),)


@dataclass(frozen=True, slots=True)
class GlobalVisualAdapter:
    """Declare one ordinary full-image adapter use."""

    lora_name: str
    strength: float


@dataclass(frozen=True, slots=True)
class SdxlVisualCase:
    """Declare one complete user-visible SDXL regional-LoRA scenario."""

    case_id: str
    label: str
    base_positive_l: str = BASE_POSITIVE_L
    base_positive_g: str = BASE_POSITIVE_G
    base_negative_l: str = BASE_NEGATIVE_L
    base_negative_g: str = BASE_NEGATIVE_G
    left_l: str = LEFT_BASE_L
    left_g: str = LEFT_BASE_G
    right_l: str = RIGHT_BASE_L
    right_g: str = RIGHT_BASE_G
    left_negative_l: str = ""
    left_negative_g: str = ""
    right_negative_l: str = ""
    right_negative_g: str = ""
    global_style_l: str = ""
    global_style_g: str = ""
    global_adapters: tuple[GlobalVisualAdapter, ...] = ()
    left_adapters: tuple[RegionalVisualAdapter, ...] = ()
    right_adapters: tuple[RegionalVisualAdapter, ...] = ()
    regional_prompt_start_percent: float = 0.0
    regional_prompt_weight: float = REGIONAL_PROMPT_WEIGHT
    mask_profile: VisualMaskProfile = VisualMaskProfile.HARD
    region_mask_feather: int = 0
    modes: tuple[VisualMode, ...] = (VisualMode.FULL,)
