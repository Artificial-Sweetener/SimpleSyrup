# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose inversion resolution and integration controls as sampler options."""

from __future__ import annotations

from typing import Any

from ..domain.sampler_options import SamplerOptions, append_sampler_capability
from .sampler_options_schema import (
    COMFY_IO,
    OptionsNodeBase,
    inversion_from_controls,
    noise_inversion_inputs,
    options_input,
    options_output,
)


class NoiseInversionOptionsV3(OptionsNodeBase):
    """Add source-derived starting noise to an immutable sampler options chain."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the accepted recipe with independently editable controls."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.NoiseInversionOptions",
            display_name="Noise Inversion Options",
            category="SimpleSyrup/Sampling/Options",
            description=(
                "Derives starting noise from an input image before sampling; "
                "control inversion quality and cost independently."
            ),
            inputs=[options_input(COMFY_IO), *noise_inversion_inputs(COMFY_IO)],
            outputs=[options_output(COMFY_IO)],
        )

    @classmethod
    def execute(
        cls,
        inversion_method: str = "euler",
        inversion_resolution_scale: float = 0.5,
        inversion_steps: int = 2,
        inversion_switch_fraction: float = 0.75,
        inversion_finishing_steps: int = 1,
        options: SamplerOptions | None = None,
    ) -> tuple[SamplerOptions]:
        """Append inversion or pass through at zero steps without preparing a model."""
        inversion = inversion_from_controls(
            inversion_method=inversion_method,
            inversion_resolution_scale=inversion_resolution_scale,
            inversion_steps=inversion_steps,
            inversion_switch_fraction=inversion_switch_fraction,
            inversion_finishing_steps=inversion_finishing_steps,
        )
        return (append_sampler_capability(options, inversion),)
