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
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
import torch.nn.functional as functional

from ..domain.contextual_diffusion import SpatialContext
from ..domain.tiled_diffusion import (
    LatentTile,
    TiledDiffusionPlan,
    gaussian_tile_weights,
    validate_tiled_diffusion_mode,
)

Latent: TypeAlias = dict[str, Any]
ApplyModel: TypeAlias = Callable[..., torch.Tensor]
ModelFunctionWrapper: TypeAlias = Callable[[ApplyModel, dict[str, Any]], torch.Tensor]
TileEvaluator: TypeAlias = Callable[[dict[str, Any]], torch.Tensor]

UNSUPPORTED_CONDITIONING_KEYS = frozenset({"area", "control", "gligen"})
CONTEXT_INVARIANT_CONDITIONING_KEYS = frozenset({"ref_latents"})


@dataclass(frozen=True)
class CachedTileWeights:
    """Store model-output and float32 accumulation weights for one tensor layout."""

    model: tuple[torch.Tensor, ...]
    accumulation: tuple[torch.Tensor, ...]


class SemanticTileWeightCache:
    """Keep semantic tile weights resident on the active sampling device."""

    def __init__(self, tiles: Sequence[LatentTile]) -> None:
        """Create an empty cache associated with one immutable tile plan."""

        self._tiles = tuple(tiles)
        self._tile_indexes = {id(tile): index for index, tile in enumerate(self._tiles)}
        self._cache_key: tuple[torch.device, torch.dtype, int] | None = None
        self._weights: CachedTileWeights | None = None

    def for_output(self, output: torch.Tensor) -> CachedTileWeights:
        """Return tile weights shaped and typed for one model output tensor."""

        cache_key = (output.device, output.dtype, output.ndim)
        if self._cache_key == cache_key and self._weights is not None:
            return self._weights
        output_weights: list[torch.Tensor] = []
        accumulation_weights: list[torch.Tensor] = []
        for tile in self._tiles:
            shape = (1,) * (output.ndim - 2) + (tile.height, tile.width)
            if tile.weight_mask is None:
                accumulation_weight = torch.ones(
                    shape,
                    device=output.device,
                    dtype=torch.float32,
                )
            else:
                if tuple(tile.weight_mask.shape) != (tile.height, tile.width):
                    raise ValueError(
                        "Semantic tile weight must match its tile dimensions."
                    )
                accumulation_weight = tile.weight_mask.to(
                    device=output.device,
                    dtype=torch.float32,
                ).reshape(shape)
            accumulation_weights.append(accumulation_weight)
            output_weights.append(accumulation_weight.to(dtype=output.dtype))
        self._cache_key = cache_key
        self._weights = CachedTileWeights(
            model=tuple(output_weights),
            accumulation=tuple(accumulation_weights),
        )
        return self._weights

    def for_tile(
        self,
        weights: CachedTileWeights,
        tile: LatentTile,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached model and accumulation weights for one planned tile."""

        index = self._tile_indexes.get(id(tile))
        if index is None:
            raise ValueError("Semantic tile weight cache received an unknown tile.")
        return weights.model[index], weights.accumulation[index]


class TileBlendWeightCache:
    """Resolve authoritative overlap weights for either tiled diffusion policy."""

    def __init__(self, plan: TiledDiffusionPlan, diffusion_mode: str) -> None:
        """Create weight caches for one immutable tiled prediction plan."""

        validate_tiled_diffusion_mode(diffusion_mode)
        self._diffusion_mode = diffusion_mode
        self._semantic_weights = SemanticTileWeightCache(plan.tiles)
        self._gaussian_weights: dict[
            tuple[torch.device, torch.dtype, int, int, int], torch.Tensor
        ] = {}

    def for_tile(
        self,
        output: torch.Tensor,
        tile: LatentTile,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return model and accumulation weights for one predicted tile."""

        semantic_weights = self._semantic_weights.for_output(output)
        model_weight, accumulation_weight = self._semantic_weights.for_tile(
            semantic_weights,
            tile,
        )
        if self._diffusion_mode == "multidiffusion":
            return model_weight, accumulation_weight

        gaussian_weight = self._gaussian_for(output, tile)
        return (
            model_weight * gaussian_weight,
            accumulation_weight * gaussian_weight.to(dtype=torch.float32),
        )

    def _gaussian_for(
        self,
        output: torch.Tensor,
        tile: LatentTile,
    ) -> torch.Tensor:
        """Return cached Mixture of Diffusers weights for one tile shape."""

        cache_key = (
            output.device,
            output.dtype,
            output.ndim,
            tile.width,
            tile.height,
        )
        cached = self._gaussian_weights.get(cache_key)
        if cached is not None:
            return cached
        weights = gaussian_tile_weights(
            tile.width,
            tile.height,
            device=output.device,
            dtype=output.dtype,
        ).reshape((1,) * (output.ndim - 2) + (tile.height, tile.width))
        self._gaussian_weights[cache_key] = weights
        return weights


class TilePredictionAccumulator:
    """Evaluate tiled model views and combine them with one selected policy."""

    def __init__(self, plan: TiledDiffusionPlan, *, diffusion_mode: str) -> None:
        """Bind an immutable plan to its overlap weighting policy."""

        self._plan = plan
        self._blend_weights = TileBlendWeightCache(plan, diffusion_mode)

    def predict(
        self,
        *,
        args: dict[str, Any],
        x: torch.Tensor,
        evaluate: TileEvaluator,
    ) -> torch.Tensor:
        """Return one canvas prediction accumulated from bounded model views."""

        output_buffer = torch.zeros_like(x)
        weight_buffer = new_spatial_weight_buffer(x, self._plan)
        input_batch_size = int(x.shape[0])
        for batch in self._plan.batches:
            tiled_args = make_tiled_model_args(
                args=args,
                tiles=batch,
                input_batch_size=input_batch_size,
                latent_height=self._plan.latent_height,
                latent_width=self._plan.latent_width,
            )
            tile_output = evaluate(tiled_args)
            for index, tile in enumerate(batch):
                tile_slice = spatial_tile_slicer(tile, x.ndim)
                start = index * input_batch_size
                end = start + input_batch_size
                model_weight, accumulation_weight = self._blend_weights.for_tile(
                    tile_output,
                    tile,
                )
                output_buffer[tile_slice] += tile_output[start:end] * model_weight
                weight_buffer[tile_slice] += accumulation_weight
        return output_buffer / weight_buffer.to(dtype=output_buffer.dtype)


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
    allow_full_context_masks: bool = False,
) -> None:
    """Reject conditioning that the selected tiled path cannot preserve."""

    if contains_unsupported_conditioning_key(
        conditioning,
        allow_full_context_masks=allow_full_context_masks,
    ):
        raise ValueError(
            f"{sampler_label} does not support regional conditioning or "
            "ControlNet in the first implementation."
        )


def contains_unsupported_conditioning_key(
    value: object,
    *,
    allow_full_context_masks: bool = False,
) -> bool:
    """Return whether nested conditioning exceeds the tiled support policy."""

    if isinstance(value, dict):
        if any(key in UNSUPPORTED_CONDITIONING_KEYS for key in value):
            return True
        if "mask" in value and (
            not allow_full_context_masks or value.get("set_area_to_bounds") is not False
        ):
            return True
        return any(
            contains_unsupported_conditioning_key(
                item,
                allow_full_context_masks=allow_full_context_masks,
            )
            for item in value.values()
        )
    if isinstance(value, list | tuple):
        return any(
            contains_unsupported_conditioning_key(
                item,
                allow_full_context_masks=allow_full_context_masks,
            )
            for item in value
        )
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


def make_spatial_context_model_args(
    *,
    args: dict[str, Any],
    contexts: Sequence[SpatialContext],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
) -> dict[str, Any]:
    """Create apply-model arguments for equally shaped spatial contexts."""

    if not contexts:
        raise ValueError("Spatial model arguments require at least one context.")
    target_shape = (contexts[0].context_height, contexts[0].context_width)
    if any(
        (context.context_height, context.context_width) != target_shape
        for context in contexts
    ):
        raise ValueError("Batched spatial contexts must use one model context shape.")
    x = args["input"]
    timestep = args["timestep"]
    conditioning = args.get("c", {})
    if not isinstance(x, torch.Tensor):
        raise ValueError("contextual sampler model input must be a tensor.")
    if not isinstance(timestep, torch.Tensor):
        raise ValueError("contextual sampler timestep must be a tensor.")
    if not isinstance(conditioning, dict):
        raise ValueError("contextual sampler conditioning must be a dict.")

    context_x = torch.cat(
        [
            resize_spatial_tensor(
                x[spatial_context_slicer(context, x.ndim)],
                height=context.context_height,
                width=context.context_width,
                mode="nearest-exact",
            )
            for context in contexts
        ],
        dim=0,
    )
    context_timestep = torch.cat([timestep] * len(contexts), dim=0)
    context_conditioning = spatial_context_conditioning(
        conditioning=conditioning,
        contexts=contexts,
        input_batch_size=input_batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        context_timestep=context_timestep,
    )
    context_args = args.copy()
    context_args["input"] = context_x
    context_args["timestep"] = context_timestep
    context_args["c"] = context_conditioning
    if "cond_or_uncond" in args:
        context_args["cond_or_uncond"] = repeat_sequence(
            args["cond_or_uncond"],
            len(contexts),
        )
    return context_args


def spatial_context_conditioning(
    *,
    conditioning: dict[str, Any],
    contexts: Sequence[SpatialContext],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
    context_timestep: torch.Tensor,
) -> dict[str, Any]:
    """Resize spatial conditioning alongside arbitrary latent contexts."""

    transformed: dict[str, Any] = {}
    for key, value in conditioning.items():
        if key == "transformer_options" and isinstance(value, dict):
            transformed[key] = tile_transformer_options(
                value,
                tile_count=len(contexts),
                tiled_timestep=context_timestep,
            )
            continue
        if key in CONTEXT_INVARIANT_CONDITIONING_KEYS:
            transformed[key] = repeat_context_invariant_value(
                value,
                context_count=len(contexts),
                input_batch_size=input_batch_size,
                conditioning_key=key,
            )
            continue
        transformed[key] = spatial_context_value(
            value,
            contexts=contexts,
            input_batch_size=input_batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
        )
    return transformed


def spatial_context_value(
    value: Any,
    *,
    contexts: Sequence[SpatialContext],
    input_batch_size: int,
    latent_height: int,
    latent_width: int,
) -> Any:
    """Transform tensors nested inside one spatial-context conditioning value."""

    if isinstance(value, torch.Tensor):
        if value.ndim >= 4 and value.shape[-2:] == (
            latent_height,
            latent_width,
        ):
            return torch.cat(
                [
                    resize_spatial_tensor(
                        value[spatial_context_slicer(context, value.ndim)],
                        height=context.context_height,
                        width=context.context_width,
                        mode="nearest-exact",
                    )
                    for context in contexts
                ],
                dim=0,
            )
        if value.ndim >= 1 and value.shape[0] == input_batch_size:
            return torch.cat([value] * len(contexts), dim=0)
        if value.ndim >= 1 and value.shape[0] == 1:
            repeats = [input_batch_size * len(contexts)] + [1] * (value.ndim - 1)
            return value.repeat(repeats)
        return value
    if isinstance(value, list):
        return [
            spatial_context_value(
                item,
                contexts=contexts,
                input_batch_size=input_batch_size,
                latent_height=latent_height,
                latent_width=latent_width,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            spatial_context_value(
                item,
                contexts=contexts,
                input_batch_size=input_batch_size,
                latent_height=latent_height,
                latent_width=latent_width,
            )
            for item in value
        )
    return value


def spatial_context_slicer(
    context: SpatialContext,
    tensor_ndim: int,
) -> tuple[slice, ...]:
    """Return a slicer for one arbitrary spatial context rectangle."""

    return (
        (slice(None),) * (tensor_ndim - 2)
        + (slice(context.y, context.y + context.height),)
        + (slice(context.x, context.x + context.width),)
    )


def resize_spatial_tensor(
    tensor: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    """Resize only the final two axes of a 4D or singleton-depth 5D tensor."""

    if tensor.shape[-2:] == (height, width):
        return tensor
    leading_shape = tensor.shape[:-2]
    flattened = tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1])
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    resized = functional.interpolate(
        flattened,
        size=(height, width),
        mode=mode,
        align_corners=align_corners,
    )
    return resized.reshape(*leading_shape, height, width)


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
        if key in CONTEXT_INVARIANT_CONDITIONING_KEYS:
            tiled[key] = repeat_context_invariant_value(
                value,
                context_count=len(tiles),
                input_batch_size=input_batch_size,
                conditioning_key=key,
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


def repeat_context_invariant_value(
    value: Any,
    *,
    context_count: int,
    input_batch_size: int,
    conditioning_key: str,
) -> Any:
    """Repeat non-spatial conditioning without cropping its tensor contents."""

    if isinstance(value, torch.Tensor):
        if value.ndim < 1:
            raise ValueError(
                f"{conditioning_key} tensors must include a batch dimension."
            )
        if value.shape[0] == input_batch_size:
            return torch.cat([value] * context_count, dim=0)
        if value.shape[0] == 1:
            repeats = [input_batch_size * context_count] + [1] * (value.ndim - 1)
            return value.repeat(repeats)
        raise ValueError(
            f"{conditioning_key} tensor batch size must be 1 or match the model "
            f"input batch size {input_batch_size}; received {value.shape[0]}."
        )
    if isinstance(value, list):
        return [
            repeat_context_invariant_value(
                item,
                context_count=context_count,
                input_batch_size=input_batch_size,
                conditioning_key=conditioning_key,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            repeat_context_invariant_value(
                item,
                context_count=context_count,
                input_batch_size=input_batch_size,
                conditioning_key=conditioning_key,
            )
            for item in value
        )
    return value


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
