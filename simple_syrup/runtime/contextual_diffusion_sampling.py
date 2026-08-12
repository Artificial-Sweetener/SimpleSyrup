# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI runtime adapter for contextual diffusion latent sampling."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, cast

import torch

from ..domain.contextual_diffusion import (
    ContextualDiffusionControls,
    ContextualDiffusionPlan,
)
from ..domain.regional_features import (
    EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    RegionalCapabilityAdmission,
)
from ..shared.logging import get_logger
from . import sampling_samplers, sampling_schedulers
from .contextual_model_wrapper import ContextualDiffusionModelWrapper
from .model_patcher_mutations import ModelUnetWrapperMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE
from .sampling_model_types import ModelFunctionWrapper
from .tiled_sampling_validation import (
    Latent,
    reject_unsupported_conditioning,
    validate_latent_samples,
    validate_sampling_controls,
    validate_tensor_shape,
)

LOGGER = get_logger(__name__)
SAMPLER_LABEL = "Contextual Diffusion"
UNIPC_SAMPLERS = frozenset({"uni_pc", "uni_pc_bh2"})


def sample_contextual_diffusion(
    *,
    model: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent_image: Latent,
    denoise: float,
    diffusion_mode: str,
    controls: ContextualDiffusionControls,
    plan: ContextualDiffusionPlan,
    capability_admission: RegionalCapabilityAdmission = (
        EMPTY_REGIONAL_CAPABILITY_ADMISSION
    ),
) -> Latent:
    """Sample one latent through global context and one tiled prediction plan."""

    validate_sampling_controls(
        steps=steps,
        denoise=denoise,
        latent_tile_width=controls.latent_context_size,
        latent_tile_height=controls.latent_context_size,
        latent_tile_batch_size=controls.latent_context_batch_size,
    )
    controls.validate()
    if sampler_name in UNIPC_SAMPLERS:
        raise ValueError("Contextual Diffusion is not compatible with UniPC samplers.")
    reject_unsupported_conditioning(
        positive,
        sampler_label=SAMPLER_LABEL,
        capability_admission=capability_admission,
    )
    reject_unsupported_conditioning(
        negative,
        sampler_label=SAMPLER_LABEL,
        capability_admission=capability_admission,
    )

    sampler = sampling_samplers.resolve_sampler(sampler_name)
    sigmas = sampling_schedulers.calculate_sigmas(
        model=model,
        scheduler_name=scheduler,
        sampler_name=sampler_name,
        steps=steps,
        denoise=denoise,
        view=sampling_schedulers.SchedulerView(
            latent_width=controls.latent_context_size,
            latent_height=controls.latent_context_size,
        ),
    ).to(model.load_device)
    latent_samples = validate_latent_samples(latent_image, sampler_label=SAMPLER_LABEL)
    comfy_sample = _comfy_sample()
    comfy_utils = _comfy_utils()
    latent_samples = comfy_sample.fix_empty_latent_channels(
        model,
        latent_samples,
        latent_image.get("downscale_ratio_spacial", None),
    )
    validate_tensor_shape(latent_samples, sampler_label=SAMPLER_LABEL)
    if latent_samples.shape[-2:] != (plan.latent_height, plan.latent_width):
        raise ValueError(
            "Contextual Diffusion plan dimensions must match the sampled latent shape."
        )

    sampling_model = clone_model_with_contextual_diffusion(
        model,
        plan=plan,
        controls=controls,
        sigmas=sigmas,
        diffusion_mode=diffusion_mode,
    )
    batch_inds = latent_image.get("batch_index")
    noise = comfy_sample.prepare_noise(latent_samples, seed, batch_inds)
    callback = _latent_preview().prepare_callback(sampling_model, steps)
    samples = comfy_sample.sample_custom(
        sampling_model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_samples,
        noise_mask=latent_image.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy_utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )

    LOGGER.info(
        "KSampler Contextual Diffusion pass completed",
        extra={
            "operation": "ksampler_contextual_diffusion",
            "sampler": sampler_name,
            "scheduler": scheduler,
            "steps": steps,
            "denoise": denoise,
            "diffusion_mode": diffusion_mode,
            "latent_width": plan.latent_width,
            "latent_height": plan.latent_height,
            "context_size": controls.latent_context_size,
            "overlap": controls.latent_context_overlap,
            "tile_count": len(plan.tile_plan.tiles),
            "segs_guided": any(
                tile.weight_mask is not None for tile in plan.tile_plan.tiles
            ),
            "model_calls_per_prediction": (
                1
                if len(plan.tile_plan.tiles) == 1
                else len(plan.tile_plan.batches) + int(controls.global_weight > 0)
            ),
        },
    )
    output = latent_image.copy()
    output.pop("downscale_ratio_spacial", None)
    output["samples"] = samples
    return output


def clone_model_with_contextual_diffusion(
    model: Any,
    *,
    plan: ContextualDiffusionPlan,
    controls: ContextualDiffusionControls,
    sigmas: torch.Tensor,
    diffusion_mode: str,
) -> Any:
    """Derive a model with one pre-CFG contextual prediction wrapper."""

    old_wrapper = model.model_options.get("model_function_wrapper")
    if old_wrapper is not None and not callable(old_wrapper):
        raise ValueError("Existing model_function_wrapper is not callable.")
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=sigmas,
        diffusion_mode=diffusion_mode,
        existing_wrapper=cast(ModelFunctionWrapper | None, old_wrapper),
    )
    return PATCHER_LIFECYCLE.derive_model(
        model,
        (ModelUnetWrapperMutation(wrapper),),
        operation="SimpleSyrup contextual diffusion",
    )


def _comfy_sample() -> ModuleType:
    """Import ComfyUI sample helpers lazily."""

    return import_module("comfy.sample")


def _comfy_utils() -> ModuleType:
    """Import ComfyUI progress state lazily."""

    return import_module("comfy.utils")


def _latent_preview() -> ModuleType:
    """Import ComfyUI preview helpers lazily."""

    return import_module("latent_preview")
