# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Expose automatic learned and native latent scaling through one method control."""

from __future__ import annotations

from typing import Any

from ..runtime.latent_upscale_catalog import latent_model_choices
from ..services.latent_upscale_service import LatentUpscaleService
from .sampler_options_schema import COMFY_IO, OptionsNodeBase


class SimpleUpscaleLatentV3(OptionsNodeBase):
    """Resize the clean latent while retaining its source noise provenance."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare one method dropdown and a scale multiplier without loading
        weights.
        """
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.SimpleUpscaleLatent",
            display_name="Simple Upscale Latent",
            category="SimpleSyrup/Latent",
            description=(
                "Upscales clean latents with a compatible learned model "
                "or native interpolation, preserving sampling provenance."
            ),
            inputs=[
                COMFY_IO.Latent.Input(
                    "latent",
                    tooltip=(
                        "Clean latent to enlarge, carrying source sampling "
                        "provenance when noise reuse is needed."
                    ),
                ),
                COMFY_IO.Combo.Input(
                    "method",
                    options=latent_model_choices(),
                    default="Auto",
                    tooltip=(
                        "Auto selects a compatible learned model from sampling "
                        "provenance, or Comfy's bislerp for unknown latent formats."
                    ),
                ),
                COMFY_IO.Float.Input(
                    "scale_factor",
                    default=2.0,
                    min=0.01,
                    max=8.0,
                    step=0.01,
                    tooltip=(
                        "Width and height multiplier; learned models enlarge "
                        "through internal stages of at most 2x."
                    ),
                ),
            ],
            outputs=[
                COMFY_IO.Latent.Output(
                    "latent",
                    tooltip=(
                        "Resized clean latent retaining source sampling "
                        "provenance for downstream noise reuse."
                    ),
                )
            ],
        )

    @classmethod
    def execute(
        cls, latent: dict[str, Any], method: str = "Auto", scale_factor: float = 2.0
    ) -> tuple[dict[str, Any]]:
        """Delegate method selection and inference to the latent upscale service."""
        return (LatentUpscaleService().upscale(latent, method, scale_factor),)
