# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""Shared runtime helpers for tiled latent denoising samplers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import torch

from ..domain.tiled_diffusion import LatentTile, TiledDiffusionPlan

Latent: TypeAlias = dict[str, Any]
ApplyModel: TypeAlias = Callable[..., torch.Tensor]
ModelFunctionWrapper: TypeAlias = Callable[[ApplyModel, dict[str, Any]], torch.Tensor]

UNSUPPORTED_CONDITIONING_KEYS = frozenset({"area", "mask", "control", "gligen"})


def validate_sampling_controls(
    *,
    steps: int,
    denoise: float,
    latent_tile_width: int,
    latent_tile_height: int,
    latent_tile_batch_size: int,
) -> None:
    """Reject invalid KSampler and tile controls before runtime side effects."""

    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if not 0.0 <= denoise <= 1.0:
        raise ValueError("denoise must be between 0 and 1.")
    if latent_tile_width < 4:
        raise ValueError("latent_tile_width must be at least 4.")
    if latent_tile_height < 4:
        raise ValueError("latent_tile_height must be at least 4.")
    if latent_tile_batch_size < 1:
        raise ValueError("latent_tile_batch_size must be at least 1.")


def validate_latent_samples(
    latent_image: Latent,
    *,
    sampler_label: str,
) -> torch.Tensor:
    """Return validated samples from a ComfyUI latent dictionary."""

    samples = latent_image.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise ValueError("latent samples must be a torch tensor.")
    validate_tensor_shape(samples, sampler_label=sampler_label)
    return samples


def validate_tensor_shape(samples: torch.Tensor, *, sampler_label: str) -> None:
    """Reject unsupported latent tensor shapes before spatial tiling."""

    if getattr(samples, "is_nested", False):
        raise ValueError(
            f"{sampler_label} requires non-nested latent samples shaped "
            "[batch, channels, height, width] or "
            "[batch, channels, 1, height, width]."
        )
    if samples.ndim == 4:
        return
    if samples.ndim == 5 and int(samples.shape[2]) == 1:
        return
    if samples.ndim == 5:
        raise ValueError(
            f"{sampler_label} 5D latent support requires a singleton third "
            "axis shaped [batch, channels, 1, height, width]."
        )
    raise ValueError(
        f"{sampler_label} requires latent samples shaped "
        "[batch, channels, height, width] or "
        "[batch, channels, 1, height, width]."
    )


def reject_unsupported_conditioning(
    conditioning: object,
    *,
    sampler_label: str,
) -> None:
    """Reject regional and external-control conditioning for basic tiled samplers."""

    if contains_unsupported_conditioning_key(conditioning):
        raise ValueError(
            f"{sampler_label} does not support regional conditioning or "
            "ControlNet in the first implementation."
        )


def contains_unsupported_conditioning_key(value: object) -> bool:
    """Return whether a nested conditioning object contains unsupported keys."""

    if isinstance(value, dict):
        if any(key in UNSUPPORTED_CONDITIONING_KEYS for key in value):
            return True
        return any(
            contains_unsupported_conditioning_key(item) for item in value.values()
        )
    if isinstance(value, list | tuple):
        return any(contains_unsupported_conditioning_key(item) for item in value)
    return False


def spatial_tile_slicer(tile: LatentTile, tensor_ndim: int) -> tuple[slice, ...]:
    """Return a slicer that crops only a tensor's final height and width axes."""

    return (
        (slice(None),) * (tensor_ndim - 2)
        + (slice(tile.y, tile.y + tile.height),)
        + (slice(tile.x, tile.x + tile.width),)
    )


def new_spatial_weight_buffer(
    x: torch.Tensor,
    plan: TiledDiffusionPlan,
) -> torch.Tensor:
    """Create a float32 spatial weight buffer broadcastable over non-spatial axes."""

    return torch.zeros(
        (1,) * (x.ndim - 2) + (plan.latent_height, plan.latent_width),
        device=x.device,
        dtype=torch.float32,
    )


def make_tiled_model_args(
    *,
    args: dict[str, Any],
    tiles: Sequence[LatentTile],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
) -> dict[str, Any]:
    """Create apply-model args for one spatial tile batch."""

    x = args["input"]
    timestep = args["timestep"]
    conditioning = args.get("c", {})
    if not isinstance(x, torch.Tensor):
        raise ValueError("tiled sampler model input must be a tensor.")
    if not isinstance(timestep, torch.Tensor):
        raise ValueError("tiled sampler timestep must be a tensor.")
    if not isinstance(conditioning, dict):
        raise ValueError("tiled sampler conditioning must be a dict.")

    tiled_x = torch.cat(
        [x[spatial_tile_slicer(tile, x.ndim)] for tile in tiles],
        dim=0,
    )
    tiled_timestep = torch.cat([timestep] * len(tiles), dim=0)
    tiled_conditioning = tile_conditioning(
        conditioning=conditioning,
        tiles=tiles,
        input_batch_size=input_batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        tiled_timestep=tiled_timestep,
    )
    tiled_args = args.copy()
    tiled_args["input"] = tiled_x
    tiled_args["timestep"] = tiled_timestep
    tiled_args["c"] = tiled_conditioning
    if "cond_or_uncond" in args:
        tiled_args["cond_or_uncond"] = repeat_sequence(
            args["cond_or_uncond"],
            len(tiles),
        )
    return tiled_args


def tile_conditioning(
    *,
    conditioning: dict[str, Any],
    tiles: Sequence[LatentTile],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
    tiled_timestep: torch.Tensor,
) -> dict[str, Any]:
    """Tile ComfyUI apply-model conditioning alongside latent tiles."""

    tiled: dict[str, Any] = {}
    for key, value in conditioning.items():
        if key == "transformer_options" and isinstance(value, dict):
            tiled[key] = tile_transformer_options(
                value,
                tile_count=len(tiles),
                tiled_timestep=tiled_timestep,
            )
            continue
        tiled[key] = tile_value(
            value,
            tiles=tiles,
            input_batch_size=input_batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
        )
    return tiled


def tile_transformer_options(
    options: dict[str, Any],
    *,
    tile_count: int,
    tiled_timestep: torch.Tensor,
) -> dict[str, Any]:
    """Repeat transformer metadata that is aligned to the model batch."""

    tiled_options = options.copy()
    if "cond_or_uncond" in options:
        tiled_options["cond_or_uncond"] = repeat_sequence(
            options["cond_or_uncond"],
            tile_count,
        )
    if "uuids" in options:
        tiled_options["uuids"] = repeat_sequence(options["uuids"], tile_count)
    if "sigmas" in options:
        tiled_options["sigmas"] = tiled_timestep
    return tiled_options


def tile_value(
    value: Any,
    *,
    tiles: Sequence[LatentTile],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
) -> Any:
    """Tile tensors inside a conditioning value while preserving containers."""

    if isinstance(value, torch.Tensor):
        return tile_tensor(
            value,
            tiles=tiles,
            input_batch_size=input_batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
        )
    if isinstance(value, list):
        return [
            tile_value(
                item,
                tiles=tiles,
                input_batch_size=input_batch_size,
                latent_height=latent_height,
                latent_width=latent_width,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            tile_value(
                item,
                tiles=tiles,
                input_batch_size=input_batch_size,
                latent_height=latent_height,
                latent_width=latent_width,
            )
            for item in value
        )
    return value


def tile_tensor(
    tensor: torch.Tensor,
    *,
    tiles: Sequence[LatentTile],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Repeat or crop a conditioning tensor for a tile batch."""

    if tensor.ndim >= 4 and tensor.shape[-2:] == (latent_height, latent_width):
        return torch.cat(
            [
                tensor[
                    ...,
                    tile.y : tile.y + tile.height,
                    tile.x : tile.x + tile.width,
                ]
                for tile in tiles
            ],
            dim=0,
        )
    if tensor.ndim >= 1 and tensor.shape[0] == input_batch_size:
        return torch.cat([tensor] * len(tiles), dim=0)
    if tensor.ndim >= 1 and tensor.shape[0] == 1:
        repeat_shape = [input_batch_size * len(tiles)] + [1] * (tensor.ndim - 1)
        return tensor.repeat(repeat_shape)
    return tensor


def repeat_sequence(value: object, times: int) -> object:
    """Repeat a metadata sequence once per tile."""

    if isinstance(value, list):
        return value * times
    if isinstance(value, tuple):
        return value * times
    return value
