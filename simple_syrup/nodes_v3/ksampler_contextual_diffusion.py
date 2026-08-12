# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Native Comfy v3 node for Contextual Diffusion sampling."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..services.contextual_diffusion_sampling_service import (
    ContextualDiffusionSamplingService,
)
from .ksampler_schema import (
    contextual_diffusion_inputs,
    ksampler_inputs,
    optional_regional_sampling_inputs,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class KSamplerContextualDiffusionV3(_ComfyNodeBase):
    """Edit large latents through coordinated global and detailed contexts."""

    service_class: ClassVar[type[ContextualDiffusionSamplingService]] = (
        ContextualDiffusionSamplingService
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the native Contextual Diffusion KSampler schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.KSamplerContextualDiffusion",
            display_name="KSampler (Contextual Diffusion)",
            category="SimpleSyrup/Sampling",
            description=(
                "Preserves composition while applying appearance and subject-detail "
                "edits to large latents through global context and optional "
                "SEGS-guided tiles."
            ),
            search_aliases=[
                "ksampler",
                "contextual diffusion",
                "contextual tiled diffusion",
                "high resolution edit",
                "sam tiled diffusion",
            ],
            inputs=[
                *ksampler_inputs(_comfy_io, steps_default=4, cfg_default=1.0),
                *contextual_diffusion_inputs(_comfy_io),
                *optional_regional_sampling_inputs(
                    _comfy_io,
                    segs_tooltip=tooltips.CONTEXTUAL_DIFFUSION_SEGS,
                ),
            ],
            outputs=[
                _comfy_io.Latent.Output(
                    "latent",
                    tooltip=tooltips.DENOISED_LATENT_OUTPUT,
                ),
                _comfy_io.SEGS.Output(
                    "contexts_segs",
                    tooltip=tooltips.CONTEXTUAL_DIFFUSION_CONTEXTS_OUTPUT,
                ),
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
        latent_context_size: int = 96,
        latent_context_overlap: int = 32,
        latent_context_batch_size: int = 4,
        global_weight: float = 1.0,
        global_steps: int = 1,
        global_decay: float = 0.5,
        segs: object | None = None,
        region_masks: object | None = None,
        regional_prompt_weight: float = 0.5,
        region_mask_feather: int = 0,
    ) -> tuple[dict[str, Any], object]:
        """Delegate Contextual Diffusion sampling to its application service."""

        result = cls.service_class().sample(
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
            diffusion_mode=diffusion_mode,
            latent_context_size=latent_context_size,
            latent_context_overlap=latent_context_overlap,
            latent_context_batch_size=latent_context_batch_size,
            global_weight=global_weight,
            global_steps=global_steps,
            global_decay=global_decay,
            segs=segs,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
        )
        return result.latent, result.contexts
