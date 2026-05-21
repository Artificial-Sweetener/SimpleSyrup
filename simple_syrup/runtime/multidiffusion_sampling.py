# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""ComfyUI runtime adapter for MultiDiffusion tiled sampling."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from types import ModuleType
from typing import Any, cast

import torch

from ..domain.tiled_diffusion import (
    LatentTile,
    TiledDiffusionPlan,
    build_tiled_diffusion_plan,
)
from ..shared.logging import get_logger
from . import sampling_samplers, sampling_schedulers
from .detail_previews import DetailPreviewContext, prepare_detail_preview_callback
from .tiled_sampling import (
    ApplyModel,
    Latent,
    ModelFunctionWrapper,
    make_tiled_model_args,
    new_spatial_weight_buffer,
    reject_unsupported_conditioning,
    spatial_tile_slicer,
    validate_latent_samples,
    validate_sampling_controls,
    validate_tensor_shape,
)

LOGGER = get_logger(__name__)
SAMPLER_LABEL = "MultiDiffusion"
UNIPC_SAMPLERS = frozenset({"uni_pc", "uni_pc_bh2"})


def sample_multidiffusion(
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
) -> Latent:
    """Sample a latent with a cloned model patched for MultiDiffusion."""

    validate_sampling_controls(
        steps=steps,
        denoise=denoise,
        latent_tile_width=latent_tile_width,
        latent_tile_height=latent_tile_height,
        latent_tile_batch_size=latent_tile_batch_size,
    )
    _reject_unipc_sampler(sampler_name)
    reject_unsupported_conditioning(positive, sampler_label=SAMPLER_LABEL)
    reject_unsupported_conditioning(negative, sampler_label=SAMPLER_LABEL)

    sampler = sampling_samplers.resolve_sampler(sampler_name)
    sigmas = sampling_schedulers.calculate_sigmas(
        model=model,
        scheduler_name=scheduler,
        sampler_name=sampler_name,
        steps=steps,
        denoise=denoise,
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
    sampling_model, plan = clone_model_with_multidiffusion(
        model,
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=latent_tile_width,
        tile_height=latent_tile_height,
        overlap=latent_tile_overlap,
        tile_batch_size=latent_tile_batch_size,
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
        "KSampler MultiDiffusion pass completed",
        extra={
            "operation": "ksampler_multidiffusion",
            "sampler": sampler_name,
            "scheduler": scheduler,
            "steps": steps,
            "denoise": denoise,
            "latent_width": latent_width,
            "latent_height": latent_height,
            "latent_ndim": latent_samples.ndim,
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


def clone_model_with_multidiffusion(
    model: Any,
    *,
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
    tile_batch_size: int,
) -> tuple[Any, TiledDiffusionPlan]:
    """Return a model clone patched with a pre-CFG MultiDiffusion wrapper."""

    plan = build_tiled_diffusion_plan(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
    )
    cloned_model = model.clone()
    old_wrapper = cloned_model.model_options.get("model_function_wrapper")
    if old_wrapper is not None and not callable(old_wrapper):
        raise ValueError("Existing model_function_wrapper is not callable.")

    wrapper = MultiDiffusionModelWrapper(
        plan=plan,
        existing_wrapper=cast(ModelFunctionWrapper | None, old_wrapper),
    )
    cloned_model.set_model_unet_function_wrapper(wrapper)
    return cloned_model, plan


class MultiDiffusionModelWrapper:
    """Average tiled model predictions before ComfyUI CFG combines them."""

    def __init__(
        self,
        *,
        plan: TiledDiffusionPlan,
        existing_wrapper: ModelFunctionWrapper | None,
    ) -> None:
        """Create the model wrapper for one latent sampling shape."""

        self._plan = plan
        self._existing_wrapper = existing_wrapper

    def __call__(
        self,
        apply_model: ApplyModel,
        args: dict[str, Any],
    ) -> torch.Tensor:
        """Run the original model on latent tiles and average overlapping output."""

        x = args["input"]
        if not isinstance(x, torch.Tensor):
            raise ValueError("MultiDiffusion model input must be a tensor.")
        validate_tensor_shape(x, sampler_label=SAMPLER_LABEL)
        if x.shape[-2:] != (self._plan.latent_height, self._plan.latent_width):
            return self._call_original(apply_model, args)
        if len(self._plan.tiles) <= 1:
            return self._call_original(apply_model, args)

        timestep = args["timestep"]
        if not isinstance(timestep, torch.Tensor):
            raise ValueError("MultiDiffusion timestep must be a tensor.")
        conditioning = args.get("c", {})
        if not isinstance(conditioning, dict):
            raise ValueError("MultiDiffusion conditioning must be a dict.")
        if conditioning.get("control") is not None:
            raise ValueError(
                "MultiDiffusion does not support regional conditioning or "
                "ControlNet in the first implementation."
            )

        output_buffer = torch.zeros_like(x)
        weight_buffer = new_spatial_weight_buffer(x, self._plan)
        input_batch_size = int(x.shape[0])

        for batch in self._plan.batches:
            tiled_args = self._make_tiled_args(
                args=args,
                tiles=batch,
                input_batch_size=input_batch_size,
            )
            tile_output = self._call_original(apply_model, tiled_args)
            for index, tile in enumerate(batch):
                tile_slice = spatial_tile_slicer(tile, x.ndim)
                start = index * input_batch_size
                end = start + input_batch_size
                output_buffer[tile_slice] += tile_output[start:end]
                weight_buffer[tile_slice] += 1.0

        return output_buffer / weight_buffer.to(dtype=output_buffer.dtype)

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
            raise ValueError("MultiDiffusion conditioning must be a dict.")
        return apply_model(args["input"], args["timestep"], **conditioning)

    def _make_tiled_args(
        self,
        *,
        args: dict[str, Any],
        tiles: Sequence[LatentTile],
        input_batch_size: int,
    ) -> dict[str, Any]:
        """Create apply-model args for one tile batch."""

        return make_tiled_model_args(
            args=args,
            tiles=tiles,
            input_batch_size=input_batch_size,
            latent_height=self._plan.latent_height,
            latent_width=self._plan.latent_width,
        )


def _reject_unipc_sampler(sampler_name: str) -> None:
    """Reject UniPC samplers because MultiDiffusion is incompatible with them."""

    if sampler_name in UNIPC_SAMPLERS:
        raise ValueError("MultiDiffusion is not compatible with UniPC samplers.")


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
