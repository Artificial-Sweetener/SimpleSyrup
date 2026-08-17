# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the immutable SDXL visual-proof sampling controls."""

from __future__ import annotations

from dataclasses import dataclass

MAX_COMFY_SEED = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True, slots=True)
class SdxlVisualSamplingControls:
    """Retain one authoritative sampling configuration for visual comparisons."""

    seed: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str


SDXL_VISUAL_SAMPLING = SdxlVisualSamplingControls(
    seed=7_429_113_057,
    steps=30,
    cfg=5.0,
    sampler="euler_ancestral",
    scheduler="karras",
)


def validate_sdxl_visual_seed(seed: object) -> int:
    """Return one seed accepted by Comfy's sampler schema."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("SDXL visual seed must be an integer.")
    if not 0 <= seed <= MAX_COMFY_SEED:
        raise ValueError(f"SDXL visual seed must be between 0 and {MAX_COMFY_SEED}.")
    return seed
