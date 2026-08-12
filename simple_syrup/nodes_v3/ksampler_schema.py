# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own shared native Comfy v3 KSampler input declarations."""

from __future__ import annotations

from typing import Any

from ..domain.regional_prompting import MAX_REGIONAL_PROMPT_WEIGHT
from ..domain.tiled_diffusion import TILED_DIFFUSION_MODES
from ..nodes import tooltips
from ..runtime import sampling_samplers, sampling_schedulers

MAX_LATENT_TILE_SIZE = 512
MAX_LATENT_CONTEXT_SIZE = 512


def ksampler_inputs(
    comfy_io: Any,
    *,
    steps_default: int,
    cfg_default: float,
) -> list[Any]:
    """Return standard KSampler inputs with caller-selected defaults."""

    conditioning = comfy_io.Custom("CONDITIONING,CONDITIONING_BATCH")
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
            default=steps_default,
            min=1,
            max=10000,
            tooltip=tooltips.SAMPLING_STEPS,
        ),
        comfy_io.Float.Input(
            "cfg",
            default=cfg_default,
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
        conditioning.Input("positive", tooltip=tooltips.POSITIVE_CONDITIONING),
        conditioning.Input("negative", tooltip=tooltips.NEGATIVE_CONDITIONING),
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


def tiled_diffusion_inputs(comfy_io: Any) -> list[Any]:
    """Return tiled diffusion mode and latent tile controls."""

    return [
        comfy_io.Combo.Input(
            "diffusion_mode",
            options=list(TILED_DIFFUSION_MODES),
            default="multidiffusion",
            tooltip=tooltips.TILED_DIFFUSION_MODE,
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


def contextual_diffusion_inputs(comfy_io: Any) -> list[Any]:
    """Return Contextual Diffusion layout and global schedule controls."""

    return [
        comfy_io.Combo.Input(
            "diffusion_mode",
            options=list(TILED_DIFFUSION_MODES),
            default="multidiffusion",
            tooltip=tooltips.TILED_DIFFUSION_MODE,
        ),
        comfy_io.Int.Input(
            "latent_context_size",
            default=96,
            min=16,
            max=MAX_LATENT_CONTEXT_SIZE,
            step=16,
            tooltip=tooltips.LATENT_CONTEXT_SIZE,
        ),
        comfy_io.Int.Input(
            "latent_context_overlap",
            default=32,
            min=0,
            max=256,
            step=4,
            tooltip=tooltips.LATENT_CONTEXT_OVERLAP,
        ),
        comfy_io.Int.Input(
            "latent_context_batch_size",
            default=4,
            min=1,
            max=8,
            step=1,
            tooltip=tooltips.LATENT_CONTEXT_BATCH_SIZE,
        ),
        comfy_io.Float.Input(
            "global_weight",
            default=1.0,
            min=0.0,
            max=2.0,
            step=0.05,
            tooltip=tooltips.GLOBAL_CONTEXT_WEIGHT,
        ),
        comfy_io.Int.Input(
            "global_steps",
            default=1,
            min=0,
            max=10000,
            step=1,
            tooltip=tooltips.GLOBAL_CONTEXT_STEPS,
        ),
        comfy_io.Float.Input(
            "global_decay",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            tooltip=tooltips.GLOBAL_CONTEXT_DECAY,
        ),
    ]


def optional_regional_sampling_inputs(
    comfy_io: Any,
    *,
    segs_tooltip: str,
) -> list[Any]:
    """Return optional SEGS and Regional Conditioning controls."""

    return [
        comfy_io.SEGS.Input(
            "segs",
            optional=True,
            tooltip=segs_tooltip,
        ),
        comfy_io.Mask.Input(
            "region_masks",
            optional=True,
            tooltip=tooltips.OPTIONAL_REGIONAL_MASKS,
        ),
        comfy_io.Float.Input(
            "regional_prompt_weight",
            default=0.5,
            min=0.0,
            max=MAX_REGIONAL_PROMPT_WEIGHT,
            step=0.01,
            round=0.01,
            optional=True,
            tooltip=tooltips.OPTIONAL_REGIONAL_PROMPT_WEIGHT,
        ),
        comfy_io.Int.Input(
            "region_mask_feather",
            default=0,
            min=0,
            max=512,
            step=1,
            optional=True,
            tooltip=tooltips.OPTIONAL_REGION_MASK_FEATHER,
        ),
    ]


def regional_ksampler_inputs(comfy_io: Any) -> list[Any]:
    """Return common regional KSampler inputs in workflow order."""

    base = ksampler_inputs(comfy_io, steps_default=20, cfg_default=8.0)
    return [
        *base[:6],
        *regional_conditioning_inputs(comfy_io),
        *base[8:],
    ]


def attention_coupling_ksampler_inputs(comfy_io: Any) -> list[Any]:
    """Return full-context Attention Coupling inputs with LoRA-specific guidance."""

    base = ksampler_inputs(comfy_io, steps_default=20, cfg_default=8.0)
    conditioning_batch = comfy_io.Custom("CONDITIONING_BATCH")
    return [
        comfy_io.Model.Input(
            "model",
            tooltip=(
                "Supported Anima or standard SD/SDXL model used for one shared "
                "denoiser trajectory; apply global model LoRAs before connecting it."
            ),
        ),
        *base[1:6],
        comfy_io.MultiType.Input(
            "positive",
            [comfy_io.Conditioning, conditioning_batch],
            tooltip=(
                "Global-first positive conditioning: entry 0 is global and later "
                "entries pair with masks. Regional Prompt Control WeightHooks may "
                "contain ordered full-rank Anima LoRA stacks with independent "
                "schedules; standard SD/SDXL rejects regional model-side hooks."
            ),
        ),
        comfy_io.MultiType.Input(
            "negative",
            [comfy_io.Conditioning, conditioning_batch],
            tooltip=(
                "Global-first negative conditioning aligned to the same masks; "
                "Anima regional LoRA hooks retain their negative-branch ownership "
                "and independent schedules."
            ),
        ),
        comfy_io.Mask.Input(
            "region_masks",
            tooltip=(
                "Ordered masks paired with conditioning entries 1 onward. In "
                "overlaps, prompt contributions are normalized while Anima regional "
                "LoRA deltas add in declared adapter and region order."
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
                "Balances regional cross-attention against the global prompt from "
                "0 (global only) to 1 (regional only inside solid masks); regional "
                "Anima LoRA strength remains controlled by each hook."
            ),
        ),
        comfy_io.Int.Input(
            "region_mask_feather",
            default=0,
            min=0,
            max=512,
            step=1,
            tooltip=(
                "Softens Attention Coupling and Anima regional LoRA boundaries by "
                "this many image pixels; 0 preserves authored mask values."
            ),
        ),
        *base[8:],
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
