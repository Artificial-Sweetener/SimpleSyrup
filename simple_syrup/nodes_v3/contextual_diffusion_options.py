# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configure complete contextual sampling with one square local context plan."""

from __future__ import annotations

from typing import Any

from ..domain.sampler_options import (
    ContextualDiffusionOptions,
    SamplerOptions,
    append_sampler_capability,
)
from .ksampler_schema import contextual_diffusion_inputs
from .sampler_options_schema import (
    COMFY_IO,
    OptionsNodeBase,
    options_input,
    options_output,
)


class ContextualDiffusionOptionsV3(OptionsNodeBase):
    """Schedule global scene authority over one local tile prediction."""

    @classmethod
    def define_schema(cls) -> Any:
        """Append local controls after existing widgets to preserve saved values."""
        controls = {
            control.id: control for control in contextual_diffusion_inputs(COMFY_IO)
        }
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.ContextualDiffusionOptions",
            display_name="Contextual Diffusion Options",
            category="SimpleSyrup/Sampling/Options",
            description=(
                "Samples local contexts with global scene guidance; "
                "takes precedence over connected Tiling Options."
            ),
            inputs=[
                options_input(COMFY_IO),
                controls["latent_context_size"],
                controls["global_weight"],
                controls["global_steps"],
                controls["global_decay"],
                controls["diffusion_mode"],
                controls["latent_context_overlap"],
                controls["latent_context_batch_size"],
                COMFY_IO.Boolean.Input(
                    "differential_diffusion",
                    default=False,
                    tooltip=(
                        "Uses the noise mask to vary denoising strength spatially; "
                        "preserves existing model mask behavior."
                    ),
                ),
            ],
            outputs=[options_output(COMFY_IO)],
        )

    @classmethod
    def execute(
        cls,
        latent_context_size: int = 96,
        global_weight: float = 1.0,
        global_steps: int = 1,
        global_decay: float = 0.5,
        options: SamplerOptions | None = None,
        diffusion_mode: str = "multidiffusion",
        latent_context_overlap: int = 32,
        latent_context_batch_size: int = 4,
        differential_diffusion: bool = False,
    ) -> tuple[SamplerOptions]:
        """Append complete context settings without inheriting a Tiling contribution."""
        return (
            append_sampler_capability(
                options,
                ContextualDiffusionOptions(
                    context_size=latent_context_size,
                    global_weight=global_weight,
                    global_steps=global_steps,
                    global_decay=global_decay,
                    diffusion_mode=diffusion_mode,
                    overlap=latent_context_overlap,
                    batch_size=latent_context_batch_size,
                    differential_diffusion=differential_diffusion,
                ),
            ),
        )
