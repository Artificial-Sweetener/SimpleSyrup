# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Native Comfy v3 node for selectable tiled diffusion sampling."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..services.tiled_diffusion_sampling_service import TiledDiffusionSamplingService
from .ksampler_schema import (
    ksampler_inputs,
    optional_regional_sampling_inputs,
    tiled_diffusion_inputs,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class KSamplerTiledDiffusionV3(_ComfyNodeBase):
    """Sample latents with selectable tiled diffusion denoising."""

    service_class: ClassVar[type[TiledDiffusionSamplingService]] = (
        TiledDiffusionSamplingService
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the native tiled diffusion KSampler schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.KSamplerTiledDiffusion",
            display_name="KSampler (Tiled Diffusion)",
            category="SimpleSyrup/Sampling",
            description="Denoises latents with selectable tiled diffusion sampling.",
            search_aliases=[
                "ksampler",
                "sampler",
                "tiled diffusion",
                "multidiffusion",
                "multi diffusion",
                "mixture of diffusers",
            ],
            inputs=[
                *ksampler_inputs(_comfy_io, steps_default=20, cfg_default=8.0),
                *tiled_diffusion_inputs(_comfy_io),
                *optional_regional_sampling_inputs(
                    _comfy_io,
                    segs_tooltip=(
                        "Optional image regions that guide irregular tile "
                        "boundaries while preserving the configured overlap."
                    ),
                ),
            ],
            outputs=[
                _comfy_io.Latent.Output(
                    None,
                    tooltip=tooltips.DENOISED_LATENT_OUTPUT,
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
        positive: Any,
        negative: Any,
        latent_image: dict[str, Any],
        denoise: float = 1.0,
        diffusion_mode: str = "multidiffusion",
        latent_tile_width: int = 128,
        latent_tile_height: int = 128,
        latent_tile_overlap: int = 16,
        latent_tile_batch_size: int = 4,
        segs: object | None = None,
        region_masks: object | None = None,
        regional_prompt_weight: float = 0.5,
        region_mask_feather: int = 0,
    ) -> tuple[dict[str, Any]]:
        """Delegate tiled diffusion sampling to its application service."""

        output = cls.service_class().sample(
            diffusion_mode=diffusion_mode,
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
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=None,
            segs=segs,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
        )
        return (output,)
