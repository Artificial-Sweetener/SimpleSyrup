# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Blend regional predictions inside ComfyUI calc-cond-batch execution."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import Any, TypeAlias, cast

import torch

from ..domain.regional_detailing import LatentRegion
from .tiled_sampling_validation import validate_tensor_shape

SAMPLER_LABEL = "Regional MultiDiffusion"
CalcCondBatchFunction: TypeAlias = Callable[[dict[str, Any]], list[torch.Tensor]]


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
                _prepare_region_conditioning(region.positive, args=args, x=region_x),
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


def validate_regions(
    *,
    latent_width: int,
    latent_height: int,
    regions: tuple[LatentRegion, ...],
) -> None:
    """Reject regions incompatible with the current latent shape."""

    for region in regions:
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


def _prepare_region_conditioning(
    conditioning: object,
    *,
    args: dict[str, Any],
    x: torch.Tensor,
) -> list[dict[str, Any]]:
    """Convert raw Comfy CONDITIONING into sampler-ready dictionaries."""

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


def _comfy_samplers() -> ModuleType:
    """Import Comfy sampler helpers lazily."""

    return import_module("comfy.samplers")


def _comfy_sampler_helpers() -> ModuleType:
    """Import Comfy sampler conditioning helpers lazily."""

    return import_module("comfy.sampler_helpers")
