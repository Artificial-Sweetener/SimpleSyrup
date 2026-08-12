# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""ComfyUI runtime adapter for Mixture of Diffusers tiled sampling."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, cast

import torch

from ..domain.regional_features import (
    EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    RegionalCapabilityAdmission,
)
from ..domain.tiled_diffusion import (
    TiledDiffusionPlan,
    build_tiled_diffusion_plan,
)
from ..shared.logging import get_logger
from . import sampling_samplers, sampling_schedulers
from .detail_previews import DetailPreviewContext, prepare_detail_preview_callback
from .differential_diffusion import (
    differential_diffusion_mutation,
    has_denoise_mask_function,
)
from .model_patcher_mutations import ModelUnetWrapperMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE, ModelMutation
from .sampling_model_types import (
    ApplyModel,
    ModelFunctionWrapper,
)
from .tile_prediction_accumulation import TilePredictionAccumulator
from .tiled_sampling_validation import (
    Latent,
    reject_unsupported_conditioning,
    validate_latent_samples,
    validate_sampling_controls,
    validate_tensor_shape,
)

LOGGER = get_logger(__name__)
SAMPLER_LABEL = "Mixture of Diffusers"


def sample_mixture_of_diffusers(
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
    latent_tile_width: int,
    latent_tile_height: int,
    latent_tile_overlap: int,
    latent_tile_batch_size: int,
    preview_context: DetailPreviewContext | None = None,
    differential_diffusion: bool = False,
    capability_admission: RegionalCapabilityAdmission = (
        EMPTY_REGIONAL_CAPABILITY_ADMISSION
    ),
    tiled_plan: TiledDiffusionPlan | None = None,
) -> Latent:
    """Sample a latent with a cloned model patched for Mixture of Diffusers."""

    validate_sampling_controls(
        steps=steps,
        denoise=denoise,
        latent_tile_width=latent_tile_width,
        latent_tile_height=latent_tile_height,
        latent_tile_batch_size=latent_tile_batch_size,
    )
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
            latent_width=latent_tile_width,
            latent_height=latent_tile_height,
        ),
    ).to(model.load_device)

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
    latent_height = int(latent_samples.shape[-2])
    latent_width = int(latent_samples.shape[-1])
    sampling_model, plan = clone_model_with_mixture_of_diffusers(
        model,
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=latent_tile_width,
        tile_height=latent_tile_height,
        overlap=latent_tile_overlap,
        tile_batch_size=latent_tile_batch_size,
        differential_diffusion=differential_diffusion,
        tiled_plan=tiled_plan,
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
        "KSampler Mixture of Diffusers pass completed",
        extra={
            "operation": "ksampler_mixture_of_diffusers",
            "sampler": sampler_name,
            "scheduler": scheduler,
            "steps": steps,
            "denoise": denoise,
            "latent_width": latent_width,
            "latent_height": latent_height,
            "tile_width": plan.tile_width,
            "tile_height": plan.tile_height,
            "overlap": plan.overlap,
            "tile_count": len(plan.tiles),
            "requested_tile_batch_size": plan.requested_tile_batch_size,
            "tile_batch_size": plan.tile_batch_size,
        },
    )

    output = latent_image.copy()
    output.pop("downscale_ratio_spacial", None)
    output["samples"] = samples
    return output


def clone_model_with_mixture_of_diffusers(
    model: Any,
    *,
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
    tile_batch_size: int,
    differential_diffusion: bool = False,
    tiled_plan: TiledDiffusionPlan | None = None,
) -> tuple[Any, TiledDiffusionPlan]:
    """Return a derived model patched with a pre-CFG Mixture wrapper."""

    plan = tiled_plan or build_tiled_diffusion_plan(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
    )
    _validate_supplied_plan(plan, latent_width, latent_height)
    old_wrapper = model.model_options.get("model_function_wrapper")
    if old_wrapper is not None and not callable(old_wrapper):
        raise ValueError("Existing model_function_wrapper is not callable.")

    wrapper = MixtureOfDiffusersModelWrapper(
        plan=plan,
        existing_wrapper=cast(ModelFunctionWrapper | None, old_wrapper),
    )
    mutations: list[ModelMutation] = []
    if differential_diffusion and not has_denoise_mask_function(model):
        mutations.append(differential_diffusion_mutation())
    mutations.append(ModelUnetWrapperMutation(wrapper))
    derived_model = PATCHER_LIFECYCLE.derive_model(
        model,
        mutations,
        operation="SimpleSyrup Mixture of Diffusers",
    )
    return derived_model, plan


class MixtureOfDiffusersModelWrapper:
    """Blend tiled model predictions before ComfyUI CFG combines them."""

    def __init__(
        self,
        *,
        plan: TiledDiffusionPlan,
        existing_wrapper: ModelFunctionWrapper | None,
    ) -> None:
        """Create the model wrapper for one latent sampling shape."""

        self._plan = plan
        self._existing_wrapper = existing_wrapper
        self._tile_predictions = TilePredictionAccumulator(
            plan,
            diffusion_mode="mixture_of_diffusers",
        )

    def __call__(
        self,
        apply_model: ApplyModel,
        args: dict[str, Any],
    ) -> torch.Tensor:
        """Run the original model on latent tiles and blend the predictions."""

        x = args["input"]
        if not isinstance(x, torch.Tensor):
            raise ValueError("Mixture of Diffusers model input must be a tensor.")
        validate_tensor_shape(x, sampler_label=SAMPLER_LABEL)
        if x.shape[-2:] != (self._plan.latent_height, self._plan.latent_width):
            return self._call_original(apply_model, args)
        if len(self._plan.tiles) <= 1:
            return self._call_original(apply_model, args)

        timestep = args["timestep"]
        if not isinstance(timestep, torch.Tensor):
            raise ValueError("Mixture of Diffusers timestep must be a tensor.")
        conditioning = args.get("c", {})
        if not isinstance(conditioning, dict):
            raise ValueError("Mixture of Diffusers conditioning must be a dict.")
        if conditioning.get("control") is not None:
            raise ValueError(
                "Mixture of Diffusers does not support regional conditioning or "
                "ControlNet in the first implementation."
            )

        return self._tile_predictions.predict(
            args=args,
            x=x,
            evaluate=lambda tiled_args: self._call_original(apply_model, tiled_args),
        )

    def _call_original(
        self,
        apply_model: ApplyModel,
        args: dict[str, Any],
    ) -> torch.Tensor:
        """Call the preserved model wrapper or raw apply_model."""

        if self._existing_wrapper is not None:
            return self._existing_wrapper(apply_model, args)
        conditioning = args.get("c", {})
        if not isinstance(conditioning, dict):
            raise ValueError("Mixture of Diffusers conditioning must be a dict.")
        return apply_model(args["input"], args["timestep"], **conditioning)


def _validate_supplied_plan(
    plan: TiledDiffusionPlan,
    latent_width: int,
    latent_height: int,
) -> None:
    """Reject a semantic tile plan that belongs to another latent shape."""

    if (plan.latent_width, plan.latent_height) == (latent_width, latent_height):
        return
    raise ValueError(
        "Mixture of Diffusers tiled plan dimensions must match the sampled latent "
        "shape."
    )


def _comfy_sample() -> ModuleType:
    """Import ComfyUI sample helpers lazily."""

    return import_module("comfy.sample")


def _sampling_callback(
    model: Any,
    steps: int,
    preview_context: DetailPreviewContext | None,
) -> Any:
    """Return a generic or detailer-specific sampling preview callback."""

    if preview_context is None:
        return _latent_preview().prepare_callback(model, steps)
    return prepare_detail_preview_callback(model, steps, preview_context)


def _comfy_utils() -> ModuleType:
    """Import ComfyUI utility state lazily."""

    return import_module("comfy.utils")


def _latent_preview() -> ModuleType:
    """Import ComfyUI preview helpers lazily."""

    return import_module("latent_preview")
