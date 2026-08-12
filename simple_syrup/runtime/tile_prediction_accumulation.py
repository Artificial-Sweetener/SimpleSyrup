# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""Accumulate weighted tile predictions into one latent-sized model output."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from ..domain.tiled_diffusion import (
    LatentTile,
    TiledDiffusionPlan,
    gaussian_tile_weights,
    validate_tiled_diffusion_mode,
)
from .spatial_model_arguments import make_tiled_model_args
from .spatial_tensor_projection import spatial_tile_slicer

TileEvaluator: TypeAlias = Callable[[dict[str, Any]], torch.Tensor]


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


def new_spatial_weight_buffer(
    x: torch.Tensor,
    plan: TiledDiffusionPlan,
) -> torch.Tensor:
    """Create a float32 weight buffer broadcastable over non-spatial axes."""

    return torch.zeros(
        (1,) * (x.ndim - 2) + (plan.latent_height, plan.latent_width),
        device=x.device,
        dtype=torch.float32,
    )
