# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared Comfy v3 schema declarations for regional KSamplers."""

from __future__ import annotations

from typing import Any

from ..domain.regional_prompting import MAX_REGIONAL_PROMPT_WEIGHT
from ..domain.tiled_diffusion import TILED_DIFFUSION_MODES
from ..nodes import tooltips
from ..nodes.ksampler_tiled_diffusion import MAX_LATENT_TILE_SIZE
from ..runtime import sampling_samplers, sampling_schedulers


def regional_ksampler_inputs(comfy_io: Any) -> list[Any]:
    """Return common regional KSampler inputs in workflow order."""

    return [
        comfy_io.Model.Input("model", tooltip=tooltips.SAMPLING_MODEL),
        comfy_io.Int.Input(
            "seed",
            default=0,
            min=0,
            max=0xFFFFFFFFFFFFFFFF,
            control_after_generate=True,
            tooltip=tooltips.SAMPLING_SEED,
        ),
        comfy_io.Int.Input(
            "steps",
            default=20,
            min=1,
            max=10000,
            tooltip=tooltips.SAMPLING_STEPS,
        ),
        comfy_io.Float.Input(
            "cfg",
            default=8.0,
            min=0.0,
            max=100.0,
            step=0.1,
            round=0.01,
            tooltip=tooltips.SAMPLING_CFG,
        ),
        comfy_io.Combo.Input(
            "sampler_name",
            options=list(sampling_samplers.available_samplers()),
            tooltip=tooltips.SAMPLER_NAME,
        ),
        comfy_io.Combo.Input(
            "scheduler",
            options=list(sampling_schedulers.available_schedulers()),
            tooltip=tooltips.SCHEDULER,
        ),
        *regional_conditioning_inputs(comfy_io),
        comfy_io.Latent.Input("latent_image", tooltip=tooltips.LATENT_IMAGE),
        comfy_io.Float.Input(
            "denoise",
            default=1.0,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip=tooltips.DENOISE_STRENGTH,
        ),
    ]


def regional_conditioning_inputs(comfy_io: Any) -> list[Any]:
    """Return the authoritative ordered regional-composition inputs."""

    conditioning_batch = comfy_io.Custom("CONDITIONING_BATCH")
    return [
        comfy_io.MultiType.Input(
            "positive",
            [comfy_io.Conditioning, conditioning_batch],
            tooltip=(
                "Positive conditioning whose first batch entry is global and "
                "later entries pair with masks in order."
            ),
        ),
        comfy_io.MultiType.Input(
            "negative",
            [comfy_io.Conditioning, conditioning_batch],
            tooltip=(
                "Negative conditioning whose first batch entry is global and "
                "later entries pair with masks in order."
            ),
        ),
        comfy_io.Mask.Input(
            "region_masks",
            tooltip=(
                "Ordered authored masks; mask 0 pairs with conditioning batch entry 1."
            ),
        ),
        comfy_io.Float.Input(
            "regional_prompt_weight",
            default=0.5,
            min=0.0,
            max=MAX_REGIONAL_PROMPT_WEIGHT,
            step=0.01,
            round=0.01,
            tooltip=(
                "Balances regional prompts against the global prompt; 0 uses "
                "only global prompting, 1 uses only regional prompting inside "
                "solid masks, and overlaps reduce the global share further."
            ),
        ),
        comfy_io.Int.Input(
            "region_mask_feather",
            default=0,
            min=0,
            max=512,
            step=1,
            tooltip=(
                "Softens regional mask edges by this many image pixels; 0 "
                "preserves authored mask values."
            ),
        ),
    ]


def tiled_regional_inputs(comfy_io: Any) -> list[Any]:
    """Return tile controls matching KSampler Tiled Diffusion."""

    return [
        comfy_io.Combo.Input(
            "diffusion_mode",
            options=list(TILED_DIFFUSION_MODES),
            default="multidiffusion",
            tooltip=(
                "Tiled blend method; MultiDiffusion is steady while Mixture of "
                "Diffusers weights tile centers more strongly."
            ),
        ),
        comfy_io.Int.Input(
            "latent_tile_width",
            default=128,
            min=16,
            max=MAX_LATENT_TILE_SIZE,
            step=16,
            tooltip=tooltips.LATENT_TILE_WIDTH,
        ),
        comfy_io.Int.Input(
            "latent_tile_height",
            default=128,
            min=16,
            max=MAX_LATENT_TILE_SIZE,
            step=16,
            tooltip=tooltips.LATENT_TILE_HEIGHT,
        ),
        comfy_io.Int.Input(
            "latent_tile_overlap",
            default=16,
            min=0,
            max=256,
            step=4,
            tooltip=tooltips.LATENT_TILE_OVERLAP,
        ),
        comfy_io.Int.Input(
            "latent_tile_batch_size",
            default=4,
            min=1,
            max=8,
            step=1,
            tooltip=tooltips.LATENT_TILE_BATCH_SIZE,
        ),
    ]
