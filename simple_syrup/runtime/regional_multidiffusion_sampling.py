# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""ComfyUI runtime adapter for SEGS regional MultiDiffusion sampling."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, cast

from ..domain.regional_detailing import LatentRegion
from ..domain.regional_features import EMPTY_REGIONAL_CAPABILITY_ADMISSION
from ..shared.logging import get_logger
from . import sampling_samplers, sampling_schedulers
from .detail_previews import DetailPreviewContext, prepare_detail_preview_callback
from .differential_diffusion import (
    differential_diffusion_mutation,
    has_denoise_mask_function,
)
from .model_patcher_mutations import ModelCalcCondBatchMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE, ModelMutation
from .regional_multidiffusion_prediction import (
    CalcCondBatchFunction,
    RegionalMultiDiffusionCalcCondBatch,
    validate_regions,
)
from .tiled_sampling_validation import (
    Latent,
    reject_unsupported_conditioning,
    validate_latent_samples,
    validate_tensor_shape,
)

LOGGER = get_logger(__name__)
SAMPLER_LABEL = "Regional MultiDiffusion"
UNIPC_SAMPLERS = frozenset({"uni_pc", "uni_pc_bh2"})


@dataclass(frozen=True)
class RegionalMultiDiffusionSummary:
    """Describe regional work installed on a cloned model."""

    latent_width: int
    latent_height: int
    latent_ndim: int
    region_count: int
    max_region_width: int
    max_region_height: int


def sample_regional_multidiffusion(
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
    regions: tuple[LatentRegion, ...],
    denoise: float,
    global_prompt_weight: float,
    preview_context: DetailPreviewContext | None = None,
    differential_diffusion: bool = False,
) -> Latent:
    """Sample a latent with regional MultiDiffusion prompt blending."""

    _validate_sampling_controls(
        steps=steps,
        denoise=denoise,
        global_prompt_weight=global_prompt_weight,
    )
    _reject_unipc_sampler(sampler_name)
    if not regions:
        raise ValueError("Regional MultiDiffusion requires at least one region.")
    reject_unsupported_conditioning(
        positive,
        sampler_label=SAMPLER_LABEL,
        capability_admission=EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    )
    reject_unsupported_conditioning(
        negative,
        sampler_label=SAMPLER_LABEL,
        capability_admission=EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    )
    for region in regions:
        reject_unsupported_conditioning(
            region.positive,
            sampler_label=SAMPLER_LABEL,
            capability_admission=EMPTY_REGIONAL_CAPABILITY_ADMISSION,
        )

    sampler = sampling_samplers.resolve_sampler(sampler_name)
    latent_samples = validate_latent_samples(
        latent_image,
        sampler_label=SAMPLER_LABEL,
    )
    comfy_sample = _comfy_sample()
    comfy_utils = _comfy_utils()

    latent_samples = comfy_sample.fix_empty_latent_channels(
        model,
        latent_samples,
        latent_image.get("downscale_ratio_spacial", None),
    )
    validate_tensor_shape(latent_samples, sampler_label=SAMPLER_LABEL)
    sigmas = sampling_schedulers.calculate_sigmas(
        model=model,
        scheduler_name=scheduler,
        sampler_name=sampler_name,
        steps=steps,
        denoise=denoise,
        view=sampling_schedulers.SchedulerView.from_tensor(latent_samples),
    ).to(model.load_device)
    latent_height = int(latent_samples.shape[-2])
    latent_width = int(latent_samples.shape[-1])
    sampling_model, summary = clone_model_with_regional_multidiffusion(
        model,
        latent_width=latent_width,
        latent_height=latent_height,
        latent_ndim=latent_samples.ndim,
        regions=regions,
        global_prompt_weight=global_prompt_weight,
        differential_diffusion=differential_diffusion,
    )

    batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
    noise = comfy_sample.prepare_noise(latent_samples, seed, batch_inds)
    noise_mask = latent_image.get("noise_mask", None)
    callback = _sampling_callback(sampling_model, steps, preview_context)
    samples = comfy_sample.sample_custom(
        sampling_model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_samples,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=not comfy_utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )

    LOGGER.info(
        "Regional MultiDiffusion pass completed",
        extra={
            "operation": "regional_multidiffusion",
            "sampler": sampler_name,
            "scheduler": scheduler,
            "steps": steps,
            "denoise": denoise,
            "latent_width": summary.latent_width,
            "latent_height": summary.latent_height,
            "latent_ndim": summary.latent_ndim,
            "region_count": summary.region_count,
            "max_region_width": summary.max_region_width,
            "max_region_height": summary.max_region_height,
            "region_positive_count": len(regions),
            "global_prompt_weight": global_prompt_weight,
        },
    )

    output = latent_image.copy()
    output.pop("downscale_ratio_spacial", None)
    output["samples"] = samples
    return output


def clone_model_with_regional_multidiffusion(
    model: Any,
    *,
    latent_width: int,
    latent_height: int,
    latent_ndim: int,
    regions: tuple[LatentRegion, ...],
    global_prompt_weight: float = 0.0,
    differential_diffusion: bool = False,
) -> tuple[Any, RegionalMultiDiffusionSummary]:
    """Return a derived model patched with regional calc-cond-batch blending."""

    _validate_sampling_controls(
        steps=1,
        denoise=1.0,
        global_prompt_weight=global_prompt_weight,
    )
    validate_regions(
        latent_width=latent_width,
        latent_height=latent_height,
        regions=regions,
    )
    old_wrapper = model.model_options.get("sampler_calc_cond_batch_function")
    if old_wrapper is not None and not callable(old_wrapper):
        raise ValueError("Existing sampler_calc_cond_batch_function is not callable.")

    wrapper = RegionalMultiDiffusionCalcCondBatch(
        latent_width=latent_width,
        latent_height=latent_height,
        regions=regions,
        existing_calc_cond_batch=cast(CalcCondBatchFunction | None, old_wrapper),
        global_prompt_weight=global_prompt_weight,
    )
    mutations: list[ModelMutation] = []
    if differential_diffusion and not has_denoise_mask_function(model):
        mutations.append(differential_diffusion_mutation())
    mutations.append(ModelCalcCondBatchMutation(wrapper))
    derived_model = PATCHER_LIFECYCLE.derive_model(
        model,
        mutations,
        operation="SimpleSyrup regional MultiDiffusion",
    )
    summary = RegionalMultiDiffusionSummary(
        latent_width=latent_width,
        latent_height=latent_height,
        latent_ndim=latent_ndim,
        region_count=len(regions),
        max_region_width=max(
            (region.latent_box.width for region in regions), default=0
        ),
        max_region_height=max(
            (region.latent_box.height for region in regions),
            default=0,
        ),
    )
    return derived_model, summary


def _validate_sampling_controls(
    *,
    steps: int,
    denoise: float,
    global_prompt_weight: float,
) -> None:
    """Reject invalid sampler controls before ComfyUI runtime calls."""

    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if not 0.0 <= denoise <= 1.0:
        raise ValueError("denoise must be between 0 and 1.")
    _validate_global_prompt_weight(global_prompt_weight)


def _validate_global_prompt_weight(global_prompt_weight: float) -> None:
    """Reject global prompt weights outside the normalized blend range."""

    if not 0.0 <= global_prompt_weight <= 1.0:
        raise ValueError("global_prompt_weight must be between 0.0 and 1.0.")


def _reject_unipc_sampler(sampler_name: str) -> None:
    """Reject UniPC samplers because MultiDiffusion is incompatible with them."""

    if sampler_name in UNIPC_SAMPLERS:
        raise ValueError(
            "Regional MultiDiffusion is not compatible with UniPC samplers."
        )


def _sampling_callback(
    model: Any,
    steps: int,
    preview_context: DetailPreviewContext | None,
) -> Any:
    """Return a generic or detailer-specific sampling preview callback."""

    if preview_context is None:
        return _latent_preview().prepare_callback(model, steps)
    return prepare_detail_preview_callback(model, steps, preview_context)


def _comfy_sample() -> ModuleType:
    """Import ComfyUI sample helpers lazily."""

    return import_module("comfy.sample")


def _comfy_utils() -> ModuleType:
    """Import ComfyUI utility state lazily."""

    return import_module("comfy.utils")


def _latent_preview() -> ModuleType:
    """Import ComfyUI preview helpers lazily."""

    return import_module("latent_preview")
