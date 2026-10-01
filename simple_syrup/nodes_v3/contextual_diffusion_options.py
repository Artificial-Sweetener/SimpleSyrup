# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add whole-scene context without introducing a second local tiling engine."""

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
        """Expose global controls independently of the optional Tiling contribution."""
        controls = contextual_diffusion_inputs(COMFY_IO)
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.ContextualDiffusionOptions",
            display_name="Contextual Diffusion Options",
            category="SimpleSyrup/Sampling/Options",
            description=(
                "Adds global scene context to local tiles; Tiling options "
                "can independently configure local geometry and blending."
            ),
            inputs=[
                options_input(COMFY_IO),
                *[
                    control
                    for control in controls
                    if control.id
                    in {
                        "latent_context_size",
                        "global_weight",
                        "global_steps",
                        "global_decay",
                    }
                ],
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
    ) -> tuple[SamplerOptions]:
        """Append immutable global-context settings in any chain position."""
        return (
            append_sampler_capability(
                options,
                ContextualDiffusionOptions(
                    context_size=latent_context_size,
                    global_weight=global_weight,
                    global_steps=global_steps,
                    global_decay=global_decay,
                ),
            ),
        )
