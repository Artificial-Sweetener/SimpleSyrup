# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""ComfyUI runtime adapter for SEGS regional MultiDiffusion sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, TypeAlias, cast

import torch

from ..domain.regional_detailing import LatentRegion
from ..shared.logging import get_logger
from . import sampling_samplers, sampling_schedulers
from .detail_previews import DetailPreviewContext, prepare_detail_preview_callback
from .differential_diffusion import install_differential_diffusion
from .tiled_sampling import (
    Latent,
    reject_unsupported_conditioning,
    validate_latent_samples,
    validate_tensor_shape,
)

LOGGER = get_logger(__name__)
SAMPLER_LABEL = "Regional MultiDiffusion"
UNIPC_SAMPLERS = frozenset({"uni_pc", "uni_pc_bh2"})
CalcCondBatchFunction: TypeAlias = Callable[[dict[str, Any]], list[torch.Tensor]]


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
    reject_unsupported_conditioning(positive, sampler_label=SAMPLER_LABEL)
    reject_unsupported_conditioning(negative, sampler_label=SAMPLER_LABEL)
    for region in regions:
        reject_unsupported_conditioning(region.positive, sampler_label=SAMPLER_LABEL)

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
    """Return a model clone patched with regional calc-cond-batch blending."""

    _validate_sampling_controls(
        steps=1,
        denoise=1.0,
        global_prompt_weight=global_prompt_weight,
    )
    _validate_regions(
        latent_width=latent_width,
        latent_height=latent_height,
        regions=regions,
    )
    cloned_model = model.clone()
    if differential_diffusion:
        install_differential_diffusion(cloned_model)
    old_wrapper = cloned_model.model_options.get("sampler_calc_cond_batch_function")
    if old_wrapper is not None and not callable(old_wrapper):
        raise ValueError("Existing sampler_calc_cond_batch_function is not callable.")

    wrapper = RegionalMultiDiffusionCalcCondBatch(
        latent_width=latent_width,
        latent_height=latent_height,
        regions=regions,
        existing_calc_cond_batch=cast(CalcCondBatchFunction | None, old_wrapper),
        global_prompt_weight=global_prompt_weight,
    )
    cloned_model.set_model_sampler_calc_cond_batch_function(wrapper)
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
    return cloned_model, summary


class RegionalMultiDiffusionCalcCondBatch:
    """Blend regional condition predictions before CFG is applied."""

    def __init__(
        self,
        *,
        latent_width: int,
        latent_height: int,
        regions: tuple[LatentRegion, ...],
        existing_calc_cond_batch: CalcCondBatchFunction | None,
        global_prompt_weight: float,
    ) -> None:
        """Create a calc-cond-batch wrapper for one latent sampling shape."""

        self._latent_width = latent_width
        self._latent_height = latent_height
        self._regions = regions
        self._existing_calc_cond_batch = existing_calc_cond_batch
        self._global_prompt_weight = global_prompt_weight

    def __call__(self, args: dict[str, Any]) -> list[torch.Tensor]:
        """Return fallback predictions blended with regional predictions."""

        x = args["input"]
        if not isinstance(x, torch.Tensor):
            raise ValueError("Regional MultiDiffusion model input must be a tensor.")
        validate_tensor_shape(x, sampler_label=SAMPLER_LABEL)
        if x.shape[-2:] != (self._latent_height, self._latent_width):
            return self._call_original(args)
        if not self._regions:
            return self._call_original(args)

        timestep = args["sigma"]
        if not isinstance(timestep, torch.Tensor):
            raise ValueError("Regional MultiDiffusion sigma must be a tensor.")
        conds = args["conds"]
        if not isinstance(conds, list) or not conds:
            raise ValueError("Regional MultiDiffusion conds must be a non-empty list.")

        fallback = self._call_original(args)
        regional_buffers = [torch.zeros_like(output) for output in fallback]
        regional_weights = [_new_spatial_weight(x) for _output in fallback]
        input_batch_size = int(x.shape[0])

        for region in self._regions:
            region_slice = _region_slicer(region, x.ndim)
            region_x = x[region_slice]
            region_conds = [
                _prepare_region_conditioning(
                    region.positive,
                    args=args,
                    x=region_x,
                ),
                *conds[1:],
            ]
            region_args = args.copy()
            region_args["conds"] = region_conds
            region_args["input"] = region_x
            region_args["sigma"] = timestep
            region_outputs = self._call_original(region_args)
            self._accumulate_region_outputs(
                outputs=region_outputs,
                buffers=regional_buffers,
                weights=regional_weights,
                region=region,
                input_batch_size=input_batch_size,
            )

        return [
            _blend_prediction(
                fallback_output,
                region_output,
                region_weight,
                global_prompt_weight=self._global_prompt_weight,
            )
            for fallback_output, region_output, region_weight in zip(
                fallback,
                regional_buffers,
                regional_weights,
                strict=True,
            )
        ]

    def _call_original(self, args: dict[str, Any]) -> list[torch.Tensor]:
        """Call the previous calc-cond-batch hook or ComfyUI default."""

        clean_args = args.copy()
        clean_options = _clean_model_options(
            cast(dict[str, Any], clean_args["model_options"]),
            self._existing_calc_cond_batch,
        )
        clean_args["model_options"] = clean_options
        if self._existing_calc_cond_batch is not None:
            return self._existing_calc_cond_batch(clean_args)
        comfy_samplers = _comfy_samplers()
        return cast(
            list[torch.Tensor],
            comfy_samplers.calc_cond_batch(
                clean_args["model"],
                clean_args["conds"],
                clean_args["input"],
                clean_args["sigma"],
                clean_options,
            ),
        )

    def _accumulate_region_outputs(
        self,
        *,
        outputs: list[torch.Tensor],
        buffers: list[torch.Tensor],
        weights: list[torch.Tensor],
        region: LatentRegion,
        input_batch_size: int,
    ) -> None:
        """Accumulate one region prediction into full-latent buffers."""

        for output_index, output in enumerate(outputs[: len(buffers)]):
            region_slice = _region_slicer(region, output.ndim)
            box = region.latent_box
            mask_slice = (
                region.latent_mask[
                    box.y : box.y + box.height,
                    box.x : box.x + box.width,
                ]
                .reshape((1,) * (output.ndim - 2) + (box.height, box.width))
                .to(device=output.device, dtype=torch.float32)
            )
            buffers[output_index][region_slice] += output[
                :input_batch_size
            ] * mask_slice.to(dtype=output.dtype)
            weights[output_index][region_slice] += mask_slice


def _validate_regions(
    *,
    latent_width: int,
    latent_height: int,
    regions: tuple[LatentRegion, ...],
) -> None:
    """Reject regions incompatible with the current latent shape."""

    for region in regions:
        _validate_region(region, latent_width=latent_width, latent_height=latent_height)


def _validate_region(
    region: LatentRegion,
    *,
    latent_width: int,
    latent_height: int,
) -> None:
    """Reject regions incompatible with the current latent shape."""

    box = region.latent_box
    if region.latent_mask.shape != (latent_height, latent_width):
        raise ValueError(
            f"Regional MultiDiffusion region {region.index} ('{region.label}') "
            "latent_mask must match the full latent height and width."
        )
    if box.x < 0 or box.y < 0 or box.width < 1 or box.height < 1:
        raise ValueError(
            f"Regional MultiDiffusion region {region.index} ('{region.label}') "
            "has an invalid latent box."
        )
    if box.x + box.width > latent_width or box.y + box.height > latent_height:
        raise ValueError(
            f"Regional MultiDiffusion region {region.index} ('{region.label}') "
            "latent box must fit inside the latent."
        )


def _region_slicer(region: LatentRegion, tensor_ndim: int) -> tuple[slice, ...]:
    """Return a slicer that crops a tensor to a latent region box."""

    box = region.latent_box
    return (
        (slice(None),) * (tensor_ndim - 2)
        + (slice(box.y, box.y + box.height),)
        + (slice(box.x, box.x + box.width),)
    )


def _new_spatial_weight(x: torch.Tensor) -> torch.Tensor:
    """Create a full-latent spatial weight buffer."""

    return torch.zeros(
        (1,) * (x.ndim - 2) + (int(x.shape[-2]), int(x.shape[-1])),
        device=x.device,
        dtype=torch.float32,
    )


def _blend_prediction(
    fallback: torch.Tensor,
    regional: torch.Tensor,
    weight: torch.Tensor,
    *,
    global_prompt_weight: float,
) -> torch.Tensor:
    """Blend normalized regional predictions with global fallback predictions."""

    has_region = weight > 0
    normalized = torch.where(
        has_region,
        regional / torch.clamp(weight, min=1.0e-37).to(dtype=regional.dtype),
        regional,
    )
    coverage = torch.clamp(weight, 0.0, 1.0).to(dtype=fallback.dtype)
    regional_alpha = coverage * (1.0 - global_prompt_weight)
    blended = fallback * (1.0 - regional_alpha) + normalized * regional_alpha
    return torch.where(has_region, blended, fallback)


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


def _prepare_region_conditioning(
    conditioning: object,
    *,
    args: dict[str, Any],
    x: torch.Tensor,
) -> list[dict[str, Any]]:
    """Convert raw Comfy CONDITIONING into sampler-ready condition dictionaries."""

    if _is_processed_conditioning(conditioning):
        return cast(list[dict[str, Any]], conditioning)
    if not isinstance(conditioning, list):
        raise TypeError("Regional MultiDiffusion region_positive must be CONDITIONING.")

    sampler_helpers = _comfy_sampler_helpers()
    comfy_samplers = _comfy_samplers()
    model = args["model"]
    converted = cast(list[dict[str, Any]], sampler_helpers.convert_cond(conditioning))
    comfy_samplers.resolve_areas_and_cond_masks_multidim(
        converted,
        tuple(int(dim) for dim in x.shape[2:]),
        x.device,
    )
    comfy_samplers.calculate_start_end_timesteps(model, converted)
    if hasattr(model, "extra_conds"):
        converted = cast(
            list[dict[str, Any]],
            comfy_samplers.encode_model_conds(
                model.extra_conds,
                converted,
                x,
                x.device,
                "positive",
            ),
        )
    return converted


def _is_processed_conditioning(conditioning: object) -> bool:
    """Return whether a conditioning value is already sampler-ready."""

    if not isinstance(conditioning, list):
        return False
    if not conditioning:
        return True
    return all(
        isinstance(item, dict) and "model_conds" in item for item in conditioning
    )


def _clean_model_options(
    model_options: dict[str, Any],
    existing_calc_cond_batch: CalcCondBatchFunction | None,
) -> dict[str, Any]:
    """Return model options that cannot recurse into this wrapper."""

    clean_options = model_options.copy()
    if existing_calc_cond_batch is None:
        clean_options.pop("sampler_calc_cond_batch_function", None)
    else:
        clean_options["sampler_calc_cond_batch_function"] = existing_calc_cond_batch
    return clean_options


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


def _comfy_samplers() -> ModuleType:
    """Import ComfyUI sampler helpers lazily."""

    return import_module("comfy.samplers")


def _comfy_sampler_helpers() -> ModuleType:
    """Import ComfyUI conditioning conversion helpers lazily."""

    return import_module("comfy.sampler_helpers")


def _latent_preview() -> ModuleType:
    """Import ComfyUI preview helpers lazily."""

    return import_module("latent_preview")
