# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose one native KSampler consuming a composable capability configuration."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.sampler_options import SamplerOptions
from ..nodes import tooltips
from ..services.sampler_options_sampling_service import SamplerOptionsSamplingService
from .ksampler_schema import ksampler_inputs
from .sampler_options_schema import COMFY_IO, OptionsNodeBase, options_input


class KSamplerV3(OptionsNodeBase):
    """Execute tiling, context, inversion and attention through shared authorities."""

    service_class: ClassVar[type[SamplerOptionsSamplingService]] = (
        SamplerOptionsSamplingService
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare sampling controls, capabilities and optional spatial regions."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.KSampler",
            display_name="KSampler (SimpleSyrup)",
            category="SimpleSyrup/Sampling",
            description=(
                "Samples latents with connected sampler options for tiling, "
                "Contextual Diffusion, noise inversion and Attention Coupling."
            ),
            inputs=[
                *ksampler_inputs(COMFY_IO, steps_default=20, cfg_default=8.0),
                options_input(COMFY_IO),
                COMFY_IO.SEGS.Input(
                    "segs",
                    optional=True,
                    tooltip=(
                        "Guides local sampling regions when Tiling or Contextual "
                        "Diffusion options are connected; ignored otherwise."
                    ),
                ),
                COMFY_IO.Mask.Input(
                    "region_masks",
                    optional=True,
                    tooltip=(
                        "Ordered masks paired with global-first conditioning batches "
                        "when Attention Coupling options are connected; "
                        "ignored otherwise."
                    ),
                ),
            ],
            outputs=[
                COMFY_IO.Latent.Output(
                    "latent", tooltip=tooltips.DENOISED_LATENT_OUTPUT
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object | None = None,
        latent_image: dict[str, Any] | None = None,
        denoise: float = 1.0,
        options: SamplerOptions | None = None,
        segs: object | None = None,
        region_masks: object | None = None,
    ) -> tuple[dict[str, Any]]:
        """Delegate sampling without mutating capability configuration."""
        if latent_image is None:
            raise TypeError("KSampler requires latent_image.")
        return (
            cls.service_class().sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                options=options,
                segs=segs,
                region_masks=region_masks,
            ),
        )
