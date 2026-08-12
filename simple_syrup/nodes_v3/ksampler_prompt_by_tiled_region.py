# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 tiled KSampler for full-context regional prompting."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.regional_features import RegionalFeature, RegionalFeatureRequest
from ..nodes import tooltips
from ..services.regional_conditioning_service import RegionalConditioningService
from ..services.tiled_diffusion_sampling_service import TiledDiffusionSamplingService
from .ksampler_schema import regional_ksampler_inputs, tiled_diffusion_inputs

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class KSamplerPromptByTiledRegionV3(_ComfyNodeBase):
    """Sample tiled latents with global-first ordered regional prompts."""

    conditioning_service_class: ClassVar[type[RegionalConditioningService]] = (
        RegionalConditioningService
    )
    sampling_service_class: ClassVar[type[TiledDiffusionSamplingService]] = (
        TiledDiffusionSamplingService
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the tiled regional KSampler schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.KSamplerPromptByTiledRegion",
            display_name="KSampler (Prompt by Tiled Region)",
            category="SimpleSyrup/Sampling",
            description=(
                "Denoises large latents in overlapping tiles while preserving "
                "global and ordered mask-bound regional prompts."
            ),
            search_aliases=[
                "ksampler",
                "regional prompt",
                "tiled regional prompt",
                "regional hires fix",
            ],
            inputs=[
                *regional_ksampler_inputs(_comfy_io),
                *tiled_diffusion_inputs(_comfy_io),
            ],
            outputs=[
                _comfy_io.Latent.Output(
                    "latent",
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
        positive: object,
        negative: object,
        region_masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
        latent_image: dict[str, Any],
        denoise: float,
        diffusion_mode: str,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
    ) -> tuple[dict[str, Any]]:
        """Assemble regional conditioning and sample overlapping latent tiles."""

        assembled_positive, assembled_negative = (
            cls.conditioning_service_class().assemble(
                positive=positive,
                negative=negative,
                masks=region_masks,
                regional_prompt_weight=regional_prompt_weight,
                region_mask_feather=region_mask_feather,
            )
        )
        output = cls.sampling_service_class().sample(
            diffusion_mode=diffusion_mode,
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=assembled_positive,
            negative=assembled_negative,
            latent_image=latent_image,
            denoise=denoise,
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=None,
            feature_request=RegionalFeatureRequest(
                frozenset({RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING})
            ),
        )
        return (output,)
