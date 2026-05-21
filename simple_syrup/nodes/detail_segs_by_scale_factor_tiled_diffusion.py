# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for tiled diffusion SEGS scale-factor detailing."""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from ..domain.segs import coerce_segs_group
from ..domain.tiled_diffusion import TILED_DIFFUSION_MODES
from ..nodes import tooltips
from ..nodes.detailer_input_adapters import (
    bool_input,
    float_input,
    image_inputs,
    int_input,
    single_input,
    str_input,
    validate_image_segs_pairing,
)
from ..runtime import sampling_samplers, sampling_schedulers
from ..runtime.detail_resize import SUPPORTED_DETAIL_UPSCALE_METHODS
from ..services.detail_segs_by_scale_factor_tiled_diffusion_service import (
    DetailSEGSByScaleFactorTiledDiffusionService,
)
from .scale_factor import scale_factor_options

OPERATION = "Detail SEGS by Scale Factor w/ Tiled Diffusion"
MAX_LATENT_TILE_SIZE = 512


class DetailSEGSByScaleFactorTiledDiffusion:
    """Detail SEGS crops with scale-factor sizing and tiled diffusion."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    INPUT_IS_LIST = True
    OUTPUT_TOOLTIPS = (tooltips.DETAIL_IMAGE_OUTPUT,)
    FUNCTION = "detail"
    CATEGORY = "SimpleSyrup/Detailing"
    DESCRIPTION = "Details scaled SEGS crops with tiled diffusion sampling."
    SEARCH_ALIASES = [
        "detailer",
        "segs detailer",
        "scale factor",
        "multidiffusion detailer",
        "mixture of diffusers detailer",
        "tiled diffusion detailer",
    ]

    service_class: ClassVar[type[DetailSEGSByScaleFactorTiledDiffusionService]] = (
        DetailSEGSByScaleFactorTiledDiffusionService
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare ComfyUI inputs for tiled scale-factor detailing."""

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips.DETAIL_IMAGE}),
                "segs": ("SEGS", {"tooltip": tooltips.DETAIL_SEGS}),
                "model": ("MODEL", {"tooltip": tooltips.DETAIL_MODEL}),
                "vae": ("VAE", {"tooltip": tooltips.DETAIL_VAE}),
                "positive": (
                    "CONDITIONING,CONDITIONING_BATCH",
                    {"tooltip": tooltips.DETAIL_POSITIVE},
                ),
                "negative": (
                    "CONDITIONING,CONDITIONING_BATCH",
                    {"tooltip": tooltips.DETAIL_NEGATIVE},
                ),
                "scale_factor": (
                    "FLOAT",
                    scale_factor_options(default=1.5),
                ),
                "upscale_method": (
                    list(SUPPORTED_DETAIL_UPSCALE_METHODS),
                    {
                        "default": "lanczos",
                        "tooltip": tooltips.DETAIL_UPSCALE_METHOD,
                    },
                ),
                "clamp_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "tooltip": tooltips.DETAIL_CLAMP_SIZE,
                    },
                ),
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
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": tooltips.SAMPLING_STEPS,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
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
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": tooltips.DENOISE_STRENGTH,
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 512,
                        "tooltip": tooltips.DETAIL_FEATHER,
                    },
                ),
                "noise_mask": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": tooltips.DETAIL_NOISE_MASK,
                    },
                ),
                "noise_mask_feather": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 512,
                        "tooltip": tooltips.DETAIL_NOISE_MASK_FEATHER,
                    },
                ),
                "tiled_encode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": tooltips.DETAIL_TILED_ENCODE,
                    },
                ),
                "tiled_decode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": tooltips.DETAIL_TILED_DECODE,
                    },
                ),
                "diffusion_mode": (
                    list(TILED_DIFFUSION_MODES),
                    {
                        "default": "multidiffusion",
                        "tooltip": (
                            "Tiled sampling blend method. MultiDiffusion is steady; "
                            "Mixture of Diffusers can blend tile predictions more "
                            "softly."
                        ),
                    },
                ),
                "latent_tile_width": (
                    "INT",
                    {
                        "default": 128,
                        "min": 16,
                        "max": MAX_LATENT_TILE_SIZE,
                        "step": 16,
                        "tooltip": tooltips.LATENT_TILE_WIDTH,
                    },
                ),
                "latent_tile_height": (
                    "INT",
                    {
                        "default": 128,
                        "min": 16,
                        "max": MAX_LATENT_TILE_SIZE,
                        "step": 16,
                        "tooltip": tooltips.LATENT_TILE_HEIGHT,
                    },
                ),
                "latent_tile_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 256,
                        "step": 4,
                        "tooltip": tooltips.LATENT_TILE_OVERLAP,
                    },
                ),
                "latent_tile_batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": tooltips.LATENT_TILE_BATCH_SIZE,
                    },
                ),
            }
        }

    def detail(
        self,
        image: object,
        segs: object,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
        scale_factor: object,
        upscale_method: object,
        clamp_size: object,
        seed: object,
        steps: object,
        cfg: object,
        sampler_name: object,
        scheduler: object,
        denoise: object,
        feather: object,
        noise_mask: object,
        noise_mask_feather: object,
        tiled_encode: object,
        tiled_decode: object,
        diffusion_mode: object,
        latent_tile_width: object,
        latent_tile_height: object,
        latent_tile_overlap: object,
        latent_tile_batch_size: object,
    ) -> tuple[object]:
        """Run tiled diffusion scale-factor detailing and return the image."""

        list_mode = isinstance(image, list)
        images = image_inputs(image, OPERATION)
        segs_group = coerce_segs_group(segs)
        validate_image_segs_pairing(images, segs_group, OPERATION)

        service = self.service_class()
        outputs: list[torch.Tensor] = []
        for single_image, single_segs in zip(images, segs_group, strict=True):
            result = service.detail(
                image=single_image,
                segs=single_segs,
                model=single_input(model, "model", list_mode, OPERATION),
                vae=single_input(vae, "vae", list_mode, OPERATION),
                positive=single_input(positive, "positive", list_mode, OPERATION),
                negative=single_input(negative, "negative", list_mode, OPERATION),
                scale_factor=float_input(
                    scale_factor, "scale_factor", list_mode, OPERATION
                ),
                upscale_method=str_input(
                    upscale_method, "upscale_method", list_mode, OPERATION
                ),
                clamp_size=int_input(clamp_size, "clamp_size", list_mode, OPERATION),
                seed=int_input(seed, "seed", list_mode, OPERATION),
                steps=int_input(steps, "steps", list_mode, OPERATION),
                cfg=float_input(cfg, "cfg", list_mode, OPERATION),
                sampler_name=str_input(
                    sampler_name, "sampler_name", list_mode, OPERATION
                ),
                scheduler=str_input(scheduler, "scheduler", list_mode, OPERATION),
                denoise=float_input(denoise, "denoise", list_mode, OPERATION),
                feather=int_input(feather, "feather", list_mode, OPERATION),
                noise_mask=bool_input(noise_mask, "noise_mask", list_mode, OPERATION),
                noise_mask_feather=int_input(
                    noise_mask_feather, "noise_mask_feather", list_mode, OPERATION
                ),
                tiled_encode=bool_input(
                    tiled_encode, "tiled_encode", list_mode, OPERATION
                ),
                tiled_decode=bool_input(
                    tiled_decode, "tiled_decode", list_mode, OPERATION
                ),
                diffusion_mode=str_input(
                    diffusion_mode, "diffusion_mode", list_mode, OPERATION
                ),
                latent_tile_width=int_input(
                    latent_tile_width, "latent_tile_width", list_mode, OPERATION
                ),
                latent_tile_height=int_input(
                    latent_tile_height, "latent_tile_height", list_mode, OPERATION
                ),
                latent_tile_overlap=int_input(
                    latent_tile_overlap, "latent_tile_overlap", list_mode, OPERATION
                ),
                latent_tile_batch_size=int_input(
                    latent_tile_batch_size,
                    "latent_tile_batch_size",
                    list_mode,
                    OPERATION,
                ),
            )
            outputs.append(result.image)
        return (torch.cat(outputs, dim=0),)
