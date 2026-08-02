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
from ..shared.logging import get_logger
from . import sampling_samplers, sampling_schedulers
from .tiled_sampling import (
    ApplyModel,
    Latent,
    ModelFunctionWrapper,
    TilePredictionAccumulator,
    make_spatial_context_model_args,
    reject_unsupported_conditioning,
    resize_spatial_tensor,
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
    reject_unsupported_conditioning(positive, sampler_label=SAMPLER_LABEL)
    reject_unsupported_conditioning(negative, sampler_label=SAMPLER_LABEL)

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
    """Clone a model and install one pre-CFG contextual prediction wrapper."""

    cloned_model = model.clone()
    old_wrapper = cloned_model.model_options.get("model_function_wrapper")
    if old_wrapper is not None and not callable(old_wrapper):
        raise ValueError("Existing model_function_wrapper is not callable.")
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=sigmas,
        diffusion_mode=diffusion_mode,
        existing_wrapper=cast(ModelFunctionWrapper | None, old_wrapper),
    )
    cloned_model.set_model_unet_function_wrapper(wrapper)
    return cloned_model


class ContextualDiffusionModelWrapper:
    """Fuse one authoritative tiled prediction with scheduled global structure."""

    def __init__(
        self,
        *,
        plan: ContextualDiffusionPlan,
        controls: ContextualDiffusionControls,
        sigmas: torch.Tensor,
        diffusion_mode: str = "multidiffusion",
        existing_wrapper: ModelFunctionWrapper | None,
    ) -> None:
        """Create a wrapper for one immutable latent and context plan."""

        self._plan = plan
        self._controls = controls
        self._global_schedule = GlobalContextSchedule(
            sigmas=sigmas,
            active_steps=controls.global_steps,
            decay=controls.global_decay,
        )
        self._existing_wrapper = existing_wrapper
        self._diffusion_mode = diffusion_mode
        self._tile_predictions = TilePredictionAccumulator(
            plan.tile_plan,
            diffusion_mode=diffusion_mode,
        )

    @property
    def diffusion_mode(self) -> str:
        """Return the selected local tile blending policy."""

        return self._diffusion_mode

    def __call__(
        self,
        apply_model: ApplyModel,
        args: dict[str, Any],
    ) -> torch.Tensor:
        """Evaluate bounded contexts and return one canvas-sized prediction."""

        x = args.get("input")
        if not isinstance(x, torch.Tensor):
            raise ValueError("Contextual Diffusion model input must be a tensor.")
        validate_tensor_shape(x, sampler_label=SAMPLER_LABEL)
        if x.shape[-2:] != (self._plan.latent_height, self._plan.latent_width):
            return self._call_original(apply_model, args)
        if len(self._plan.tile_plan.tiles) == 1:
            return self._call_original(apply_model, args)

        tile_prediction = self._predict_tiles(apply_model, args, x)
        global_scale = self._global_schedule.scale_for(args.get("timestep"))
        return (
            self._apply_global_authority(
                apply_model,
                args,
                tile_prediction,
                weight=self._controls.global_weight * global_scale,
            )
            if global_scale > 0 and self._controls.global_weight > 0
            else tile_prediction
        )

    def _predict_tiles(
        self,
        apply_model: ApplyModel,
        args: dict[str, Any],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Blend the authoritative regular or SEGS-guided tiled prediction."""

        return self._tile_predictions.predict(
            args=args,
            x=x,
            evaluate=lambda tiled_args: self._call_original(apply_model, tiled_args),
        )

    def _apply_global_authority(
        self,
        apply_model: ApplyModel,
        args: dict[str, Any],
        local_prediction: torch.Tensor,
        *,
        weight: float,
    ) -> torch.Tensor:
        """Replace tile-scale scene interpretation with the whole-image prediction."""

        global_args = make_spatial_context_model_args(
            args=args,
            contexts=(self._plan.global_context,),
            input_batch_size=int(local_prediction.shape[0]),
            latent_height=self._plan.latent_height,
            latent_width=self._plan.latent_width,
        )
        global_prediction = self._call_original(apply_model, global_args)
        context = self._plan.global_context
        global_canvas = resize_spatial_tensor(
            global_prediction,
            height=self._plan.latent_height,
            width=self._plan.latent_width,
            mode="bilinear",
        )
        local_low = resize_spatial_tensor(
            local_prediction,
            height=context.context_height,
            width=context.context_width,
            mode="area",
        )
        local_low = resize_spatial_tensor(
            local_low,
            height=self._plan.latent_height,
            width=self._plan.latent_width,
            mode="bilinear",
        )
        return local_prediction + weight * (global_canvas - local_low)

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
            raise ValueError("Contextual Diffusion conditioning must be a dict.")
        return apply_model(args["input"], args["timestep"], **conditioning)


class GlobalContextSchedule:
    """Limit whole-image authority to an initial denoising-step fraction."""

    def __init__(
        self,
        *,
        sigmas: torch.Tensor,
        active_steps: int,
        decay: float,
    ) -> None:
        """Capture model-evaluation sigmas and the active initial step count."""

        step_sigmas = sigmas.detach().to(device="cpu", dtype=torch.float64).flatten()
        if step_sigmas.numel() < 2:
            raise ValueError(
                "Contextual Diffusion requires at least one denoising step."
            )
        self._step_sigmas = step_sigmas[:-1]
        self._active_steps = min(
            len(self._step_sigmas),
            max(0, active_steps),
        )
        self._decay = decay

    def scale_for(self, timestep: object) -> float:
        """Return the decayed global scale for the nearest scheduled step."""

        if self._active_steps == 0:
            return 0.0
        if not isinstance(timestep, torch.Tensor) or timestep.numel() == 0:
            raise ValueError(
                "Contextual Diffusion timestep must be a non-empty tensor."
            )
        sigma = timestep.detach().flatten()[0].to(device="cpu", dtype=torch.float64)
        step_index = int(torch.argmin(torch.abs(self._step_sigmas - sigma)).item())
        if step_index >= self._active_steps:
            return 0.0
        return self._decay**step_index


def _comfy_sample() -> ModuleType:
    """Import ComfyUI sample helpers lazily."""

    return import_module("comfy.sample")


def _comfy_utils() -> ModuleType:
    """Import ComfyUI progress state lazily."""

    return import_module("comfy.utils")


def _latent_preview() -> ModuleType:
    """Import ComfyUI preview helpers lazily."""

    return import_module("latent_preview")
