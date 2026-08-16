# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the fixed Phase 8 SDXL managed-workflow matrix."""

from __future__ import annotations

from dataclasses import dataclass

from .sampling_controls import SDXL_VISUAL_SAMPLING

SOURCE_WIDTH = 1024
SOURCE_HEIGHT = 1024
TARGET_WIDTH = 1536
TARGET_HEIGHT = 1536
UPSCALE_FACTOR = 1.5
REGION_WIDTH = round(TARGET_WIDTH * 0.4)
RIGHT_REGION_START = TARGET_WIDTH - REGION_WIDTH
REGIONAL_PROMPT_WEIGHT = 0.4
REFINEMENT_STEPS = 12
REFINEMENT_DENOISE = 0.25
TILE_SIZE = 128
TILE_OVERLAP = 64
TILE_BATCH_SIZE = 4
CONTEXTUAL_EXPECTED_MODEL_CALLS = REFINEMENT_STEPS * 5 + 1
POSITIVE_PROMPTS = (
    "cinematic wide full-body environmental scene of exactly two separate adult "
    "adventurers, one person at the far left and one person at the far right, "
    "large empty forest clearing between them, both complete bodies visible, "
    "clearly separated people, coherent perspective",
    "one complete distinct silver-haired male knight in blue armor at the far "
    "left, full body, standing alone",
    "one complete distinct red-haired female mage in crimson robes at the far "
    "right, full body, standing alone",
)
POSITIVE_PROMPTS_G = (
    "cinematic fantasy key art, coherent forest clearing, two adult adventurers, "
    "wide environmental composition, unified lighting and perspective",
    "cinematic fantasy knight character design, coherent blue armor, full body",
    "cinematic fantasy mage character design, coherent crimson robes, full body",
)
NEGATIVE_PROMPTS = (
    "single person, centered person, touching people, overlapping people, fused "
    "people, merged body, split face, half-and-half person, duplicate body, "
    "collage, split screen, cropped, low quality, blurry",
    "low quality, malformed armor, fused body",
    "low quality, malformed hands, fused body",
)
NEGATIVE_PROMPTS_G = (
    "collage, split screen, incoherent scene, duplicate bodies, fused people, "
    "cropped composition, low quality",
    "malformed knight, incoherent armor, low quality",
    "malformed mage, incoherent hands, low quality",
)


@dataclass(frozen=True, slots=True)
class SdxlIntegrationMode:
    """Describe one exact public-node execution and acceptance geometry."""

    mode_id: str
    label: str
    node_id: str
    width: int
    height: int
    expected_model_calls: int
    expected_spatial_modes: frozenset[str]


MODES = (
    SdxlIntegrationMode(
        "full",
        "SDXL full 1024 Attention Coupling",
        "SimpleSyrup.KSamplerAttentionCoupling",
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
        SDXL_VISUAL_SAMPLING.steps,
        frozenset({"full"}),
    ),
    SdxlIntegrationMode(
        "tiled-1.5x",
        "SDXL tiled 1024 to 1536 Attention Coupling",
        "SimpleSyrup.KSamplerAttentionCouplingTiled",
        TARGET_WIDTH,
        TARGET_HEIGHT,
        REFINEMENT_STEPS,
        frozenset({"tile"}),
    ),
    SdxlIntegrationMode(
        "contextual-1.5x",
        "SDXL Contextual 1024 to 1536 Attention Coupling",
        "SimpleSyrup.KSamplerAttentionCouplingContextual",
        TARGET_WIDTH,
        TARGET_HEIGHT,
        CONTEXTUAL_EXPECTED_MODEL_CALLS,
        frozenset({"tile", "contextual_global"}),
    ),
)
