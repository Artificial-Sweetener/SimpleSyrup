# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify FLUX model generations from ComfyUI's structural model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FluxGeneration(StrEnum):
    """Identify the conditioning generation used by a FLUX diffusion model."""

    FLUX = "flux"
    FLUX2 = "flux2"


class Flux2TextEncoderProfile(StrEnum):
    """Identify the text-encoder family required by a FLUX.2 architecture."""

    DEV = "dev"
    KLEIN_4B = "klein_4b"
    KLEIN_9B = "klein_9b"


@dataclass(frozen=True)
class FluxModelProfile:
    """Describe a structurally detected FLUX generation and encoder profile."""

    generation: FluxGeneration
    flux2_text_encoder: Flux2TextEncoderProfile | None = None


FLUX2_CONTEXT_DIMENSIONS: dict[int, Flux2TextEncoderProfile] = {
    15_360: Flux2TextEncoderProfile.DEV,
    7_680: Flux2TextEncoderProfile.KLEIN_4B,
    12_288: Flux2TextEncoderProfile.KLEIN_9B,
}


def classify_flux_profile(
    image_model: str | None,
    context_input_dimension: int | None,
) -> FluxModelProfile | None:
    """Return the FLUX profile represented by ComfyUI's detected dimensions."""

    if image_model == FluxGeneration.FLUX:
        return FluxModelProfile(FluxGeneration.FLUX)
    if image_model != FluxGeneration.FLUX2:
        return None
    encoder_profile = (
        FLUX2_CONTEXT_DIMENSIONS.get(context_input_dimension)
        if context_input_dimension is not None
        else None
    )
    return FluxModelProfile(FluxGeneration.FLUX2, encoder_profile)
