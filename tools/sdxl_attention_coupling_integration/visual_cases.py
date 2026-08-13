# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare the immutable U11 native-SDXL visual acceptance cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

checkpoint_a_SOURCE = Path(
    "<MODEL_ROOT>\\stable-diffusion\\Illustrious\\"
    "checkpoint_aXLIllustrious_v30WIP.safetensors"
)
character_b_SOURCE = Path(
    r"<MODEL_ROOT>\Loras\Illustrious\Character\CV.CHARACTER_B_ILXL_v1.safetensors"
)
character_c_SOURCE = Path(
    "<MODEL_ROOT>\\Loras\\Illustrious\\Character\\"
    "CHARACTER_CMR-illu-bsinky-v1.safetensors"
)
ELDEN_STYLE_SOURCE = Path(
    "<MODEL_ROOT>\\Loras\\Illustrious\\Style\\"
    "ELDEN RING Background_illustriousXL_v2.safetensors"
)

CHECKPOINT_NAME = r"simple_syrup_u11\checkpoint_a-illustrious.safetensors"
character_b_NAME = r"simple_syrup_u11\character_b-illustrious.safetensors"
character_c_NAME = r"simple_syrup_u11\character_c-illustrious.safetensors"
ELDEN_STYLE_NAME = r"simple_syrup_u11\elden-background-illustrious.safetensors"

BASE_POSITIVE_L = (
    "best quality, masterpiece, very aesthetic, official art, two mature adult "
    "women posing closely together on one velvet sofa in an ornate moonlit palace "
    "salon, interacting, shared camera, shared background, unified perspective, "
    "unified dramatic lighting, medium full shot"
)
BASE_POSITIVE_G = (
    "cohesive high quality anime key art, two adult women together in one ornate "
    "moonlit palace salon, one scene, one camera, unified lighting and perspective"
)
BASE_NEGATIVE_L = (
    "child, teen, loli, solo, one person, split screen, collage, panel boundary, "
    "separate scenes, duplicated person, fused body, merged face, extra limbs, "
    "missing limbs, cropped, blurry, low quality, text, watermark"
)
BASE_NEGATIVE_G = (
    "split composition, collage, separate backgrounds, incoherent lighting, child, "
    "duplicate bodies, fused people, low quality"
)
LEFT_BASE_L = (
    "adult woman on the left, long pink twintails, pink eyes, black sleeveless dress, "
    "black ribbons, confident smirk, leaning toward the other woman"
)
LEFT_BASE_G = "adult pink-haired woman in an elegant black dress on the left"
RIGHT_BASE_L = (
    "adult woman on the right, long black hair, blue ribbons, black cropped hoodie, "
    "black denim shorts, pink eyes, pouting, shoulder touching the other woman"
)
RIGHT_BASE_G = "adult black-haired woman in modern black clothes on the right"
character_b_L = (
    "20_character_b woman, adult woman, white hair, long hair, pale skin, pointy ears, "
    "blue eyes, vampire, fangs, red dress, sitting on the right, facing companion"
)
character_b_G = "CHARACTER_B, adult white-haired vampire woman in a red dress"
character_c_L = (
    "dfgall, adult woman, short orange hair, aqua eyes, pointy ears, grey hairband, "
    "fairy wings, elegant blue and yellow dress, sitting on the left, facing companion"
)
character_c_G = "CHARACTER_C, adult orange-haired fairy woman in an elegant dress"


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
    strength: float
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
    left_l: str = LEFT_BASE_L
    left_g: str = LEFT_BASE_G
    right_l: str = RIGHT_BASE_L
    right_g: str = RIGHT_BASE_G
    global_adapters: tuple[GlobalVisualAdapter, ...] = ()
    left_adapters: tuple[RegionalVisualAdapter, ...] = ()
    right_adapters: tuple[RegionalVisualAdapter, ...] = ()
    mask_profile: VisualMaskProfile = VisualMaskProfile.HARD
    modes: tuple[VisualMode, ...] = (VisualMode.FULL,)


def visual_cases() -> tuple[SdxlVisualCase, ...]:
    """Return the fixed twelve-case, fourteen-output acceptance matrix."""

    character_b = RegionalVisualAdapter(character_b_NAME, 0.9)
    character_c = RegionalVisualAdapter(character_c_NAME, 0.9)
    style = RegionalVisualAdapter(ELDEN_STYLE_NAME, 0.75)
    return (
        SdxlVisualCase("baseline", "No LoRA baseline"),
        SdxlVisualCase(
            "character-left",
            "CHARACTER_C character LoRA on left only",
            left_l=character_c_L,
            left_g=character_c_G,
            left_adapters=(character_c,),
        ),
        SdxlVisualCase(
            "character-right",
            "CHARACTER_B character LoRA on right only",
            right_l=character_b_L,
            right_g=character_b_G,
            right_adapters=(character_b,),
        ),
        SdxlVisualCase(
            "different-characters",
            "CHARACTER_C left and CHARACTER_B right",
            left_l=character_c_L,
            left_g=character_c_G,
            right_l=character_b_L,
            right_g=character_b_G,
            left_adapters=(character_c,),
            right_adapters=(character_b,),
        ),
        SdxlVisualCase(
            "global-style-regional-character",
            "Global Elden style with CHARACTER_B on right",
            right_l=character_b_L,
            right_g=character_b_G,
            global_adapters=(GlobalVisualAdapter(ELDEN_STYLE_NAME, 0.65),),
            right_adapters=(character_b,),
            modes=(VisualMode.FULL, VisualMode.TILED, VisualMode.CONTEXTUAL),
        ),
        SdxlVisualCase(
            "regional-style",
            "Elden style LoRA on left only",
            left_adapters=(style,),
        ),
        SdxlVisualCase(
            "same-style-global-left",
            "Same Elden style globally and stronger on left",
            global_adapters=(GlobalVisualAdapter(ELDEN_STYLE_NAME, 0.35),),
            left_adapters=(RegionalVisualAdapter(ELDEN_STYLE_NAME, 0.55),),
        ),
        SdxlVisualCase(
            "multiple-left",
            "CHARACTER_C character and Elden style on left",
            left_l=character_c_L,
            left_g=character_c_G,
            left_adapters=(character_c, RegionalVisualAdapter(ELDEN_STYLE_NAME, 0.55)),
        ),
        SdxlVisualCase(
            "same-style-both",
            "Same Elden style adapter authored in both regions",
            left_adapters=(style,),
            right_adapters=(style,),
        ),
        SdxlVisualCase(
            "scheduled-right-character",
            "CHARACTER_B right, active through 60 percent of denoising",
            right_l=character_b_L,
            right_g=character_b_G,
            right_adapters=(
                RegionalVisualAdapter(
                    character_b_NAME,
                    0.9,
                    ((0.0, 1.0), (0.6, 0.0)),
                ),
            ),
        ),
        SdxlVisualCase(
            "soft-overlap",
            "CHARACTER_C left and CHARACTER_B right with soft overlap",
            left_l=character_c_L,
            left_g=character_c_G,
            right_l=character_b_L,
            right_g=character_b_G,
            left_adapters=(character_c,),
            right_adapters=(character_b,),
            mask_profile=VisualMaskProfile.SOFT_OVERLAP,
        ),
        SdxlVisualCase(
            "uncovered-center",
            "CHARACTER_C left and CHARACTER_B right with uncovered center",
            left_l=character_c_L,
            left_g=character_c_G,
            right_l=character_b_L,
            right_g=character_b_G,
            left_adapters=(character_c,),
            right_adapters=(character_b,),
            mask_profile=VisualMaskProfile.UNCOVERED_CENTER,
        ),
    )
