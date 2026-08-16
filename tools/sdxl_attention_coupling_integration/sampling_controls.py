# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the immutable SDXL visual-proof sampling controls."""

from __future__ import annotations

from dataclasses import dataclass


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
