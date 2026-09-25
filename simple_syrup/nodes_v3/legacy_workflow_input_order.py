# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Retain persisted socket order for legacy-backed Comfy v3 nodes."""

KSAMPLER_EXTRAS_INPUT_ORDER = (
    "model",
    "seed",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "positive",
    "negative",
    "latent_image",
    "denoise",
)

DETAIL_SEGS_AS_REGIONS_INPUT_ORDER = (
    "image",
    "model",
    "vae",
    "negative",
    "positive",
    "segs",
    "region_positive",
    "global_prompt_weight",
    "scale_factor",
    "upscale_method",
    "seed",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "denoise",
    "feather",
    "noise_mask",
    "noise_mask_feather",
    "tiled_encode",
    "tiled_decode",
)

DETAIL_SEGS_BY_SCALE_FACTOR_INPUT_ORDER = (
    "image",
    "segs",
    "model",
    "vae",
    "positive",
    "negative",
    "scale_factor",
    "upscale_method",
    "clamp_size",
    "seed",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "denoise",
    "feather",
    "noise_mask",
    "noise_mask_feather",
    "tiled_encode",
    "tiled_decode",
)

DETAIL_SEGS_BY_SCALE_FACTOR_TILED_INPUT_ORDER = (
    *DETAIL_SEGS_BY_SCALE_FACTOR_INPUT_ORDER,
    "diffusion_mode",
    "latent_tile_width",
    "latent_tile_height",
    "latent_tile_overlap",
    "latent_tile_batch_size",
)
