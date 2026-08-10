# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for contextual diffusion sampling."""

from __future__ import annotations

from typing import Any, ClassVar, TypeAlias

from ..domain.regional_prompting import MAX_REGIONAL_PROMPT_WEIGHT
from ..domain.tiled_diffusion import TILED_DIFFUSION_MODES
from ..runtime import sampling_samplers, sampling_schedulers
from ..services.contextual_diffusion_sampling_service import (
    ContextualDiffusionSamplingService,
)
from . import tooltips

Latent: TypeAlias = dict[str, Any]
MAX_LATENT_CONTEXT_SIZE = 512


class KSamplerContextualDiffusion:
    """Edit large latents through coordinated global and detailed contexts."""

    RETURN_TYPES = ("LATENT", "SEGS")
    RETURN_NAMES = ("latent", "contexts_segs")
    OUTPUT_TOOLTIPS = (
        tooltips.DENOISED_LATENT_OUTPUT,
        tooltips.CONTEXTUAL_DIFFUSION_CONTEXTS_OUTPUT,
    )
    FUNCTION = "sample"
    CATEGORY = "SimpleSyrup/Sampling"
    DESCRIPTION = (
        "Preserves composition while applying appearance and subject-detail edits "
        "to large latents through global context and optional SEGS-guided tiles."
    )
    SEARCH_ALIASES = [
        "ksampler",
        "contextual diffusion",
        "contextual tiled diffusion",
        "high resolution edit",
        "sam tiled diffusion",
    ]

    service_class: ClassVar[type[ContextualDiffusionSamplingService]] = (
        ContextualDiffusionSamplingService
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare KSampler inputs and bounded contextual controls."""

        return {
            "required": {
                "model": ("MODEL", {"tooltip": tooltips.SAMPLING_MODEL}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": tooltips.SAMPLING_SEED,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 10000,
                        "tooltip": tooltips.SAMPLING_STEPS,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": tooltips.SAMPLING_CFG,
                    },
                ),
                "sampler_name": (
                    sampling_samplers.available_samplers(),
                    {"tooltip": tooltips.SAMPLER_NAME},
                ),
                "scheduler": (
                    sampling_schedulers.available_schedulers(),
                    {"tooltip": tooltips.SCHEDULER},
                ),
                "positive": (
                    "CONDITIONING,CONDITIONING_BATCH",
                    {"tooltip": tooltips.POSITIVE_CONDITIONING},
                ),
                "negative": (
                    "CONDITIONING,CONDITIONING_BATCH",
                    {"tooltip": tooltips.NEGATIVE_CONDITIONING},
                ),
                "latent_image": ("LATENT", {"tooltip": tooltips.LATENT_IMAGE}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": tooltips.DENOISE_STRENGTH,
                    },
                ),
                "diffusion_mode": (
                    list(TILED_DIFFUSION_MODES),
                    {
                        "default": "multidiffusion",
                        "tooltip": tooltips.TILED_DIFFUSION_MODE,
                    },
                ),
                "latent_context_size": (
                    "INT",
                    {
                        "default": 96,
                        "min": 16,
                        "max": MAX_LATENT_CONTEXT_SIZE,
                        "step": 16,
                        "tooltip": tooltips.LATENT_CONTEXT_SIZE,
                    },
                ),
                "latent_context_overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 256,
                        "step": 4,
                        "tooltip": tooltips.LATENT_CONTEXT_OVERLAP,
                    },
                ),
                "latent_context_batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": tooltips.LATENT_CONTEXT_BATCH_SIZE,
                    },
                ),
                "global_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": tooltips.GLOBAL_CONTEXT_WEIGHT,
                    },
                ),
                "global_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": tooltips.GLOBAL_CONTEXT_STEPS,
                    },
                ),
                "global_decay": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": tooltips.GLOBAL_CONTEXT_DECAY,
                    },
                ),
            },
            "optional": {
                "segs": (
                    "SEGS",
                    {"tooltip": tooltips.CONTEXTUAL_DIFFUSION_SEGS},
                ),
                "region_masks": (
                    "MASK",
                    {"tooltip": tooltips.OPTIONAL_REGIONAL_MASKS},
                ),
                "regional_prompt_weight": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": MAX_REGIONAL_PROMPT_WEIGHT,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": tooltips.OPTIONAL_REGIONAL_PROMPT_WEIGHT,
                    },
                ),
                "region_mask_feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": tooltips.OPTIONAL_REGION_MASK_FEATHER,
                    },
                ),
            },
        }

    def sample(
        self,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
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
    ) -> tuple[Latent, object]:
        """Delegate contextual diffusion sampling to its application service."""

        result = self.service_class().sample(
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
