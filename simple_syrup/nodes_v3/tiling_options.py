# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configure the single local tiling authority for a sampler options chain."""

from __future__ import annotations

from typing import Any

from ..domain.sampler_options import (
    SamplerOptions,
    TilingOptions,
    append_sampler_capability,
)
from .ksampler_schema import tiled_diffusion_inputs
from .sampler_options_schema import (
    COMFY_IO,
    OptionsNodeBase,
    options_input,
    options_output,
)


class TilingOptionsV3(OptionsNodeBase):
    """Add bounded local tiles, blend policy and mask-dependent denoising."""

    @classmethod
    def define_schema(cls) -> Any:
        """Share tiled controls and expose mask-dependent denoising."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.TilingOptions",
            display_name="Tiling Options",
            category="SimpleSyrup/Sampling/Options",
            description=(
                "Samples bounded local tiles; ignored when "
                "Contextual Diffusion Options is connected."
            ),
            inputs=[
                options_input(COMFY_IO),
                *tiled_diffusion_inputs(COMFY_IO),
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
        diffusion_mode: str = "multidiffusion",
        latent_tile_width: int = 128,
        latent_tile_height: int = 128,
        latent_tile_overlap: int = 16,
        latent_tile_batch_size: int = 4,
        differential_diffusion: bool = False,
        options: SamplerOptions | None = None,
    ) -> tuple[SamplerOptions]:
        """Append validated tiling without changing the incoming chain."""
        return (
            append_sampler_capability(
                options,
                TilingOptions(
                    diffusion_mode=diffusion_mode,
                    width=latent_tile_width,
                    height=latent_tile_height,
                    overlap=latent_tile_overlap,
                    batch_size=latent_tile_batch_size,
                    differential_diffusion=differential_diffusion,
                ),
            ),
        )
