# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Name generic managed aliases for external Anima benchmark fixtures."""

from __future__ import annotations

ANIMA_DIFFUSION_SELECTION = r"simple_syrup_anima\diffusion-model.safetensors"
ANIMA_TEXT_ENCODER_SELECTION = r"simple_syrup_anima\text-encoder.safetensors"
ANIMA_VAE_SELECTION = r"simple_syrup_anima\vae.safetensors"

ANIMA_BASE_SELECTIONS = (
    ("diffusion_models", ANIMA_DIFFUSION_SELECTION),
    ("text_encoders", ANIMA_TEXT_ENCODER_SELECTION),
    ("vae", ANIMA_VAE_SELECTION),
)
