"""Characterize prediction accumulation imported from the mixed tiled runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from simple_syrup.domain.spatial_views import SpatialBatchLayout, SpatialViewKind
from simple_syrup.domain.tiled_diffusion import (
    LatentTile,
    build_tiled_diffusion_plan,
    gaussian_tile_weights,
)
from simple_syrup.runtime.tile_prediction_accumulation import (
    SemanticTileWeightCache,
    TilePredictionAccumulator,
)


def _canvas(layout: str) -> torch.Tensor:
    """Return a two-item BCHW or singleton-depth BCDHW source canvas."""

    values = torch.arange(2 * 2 * 4 * 8, dtype=torch.float32).reshape((2, 2, 4, 8))
    return values if layout == "bchw" else values.unsqueeze(2)


@pytest.mark.parametrize("layout", ["bchw", "bcdhw"])
@pytest.mark.parametrize("diffusion_mode", ["multidiffusion", "mixture_of_diffusers"])
def test_accumulator_reconstructs_nonoverlap_canvas_and_batch_metadata(
    layout: str, diffusion_mode: str
) -> None:
    """Preserve tile-major batching, layouts, metadata, and exact reconstruction."""

    plan = build_tiled_diffusion_plan(8, 4, 4, 4, 0, 2)
    x = _canvas(layout)
    timestep = torch.tensor([0.5, 0.75])
    cross_attention = torch.tensor([[[1.0]], [[2.0]]])
    args: dict[str, Any] = {
        "input": x,
        "timestep": timestep,
        "cond_or_uncond": [0, 1],
        "c": {
            "c_crossattn": cross_attention,
            "transformer_options": {
                "cond_or_uncond": [0, 1],
                "uuids": ("positive", "negative"),
                "sigmas": torch.tensor([9.0, 8.0]),
            },
        },
    }
    observations: list[dict[str, Any]] = []

    def evaluate(tiled_args: dict[str, Any]) -> torch.Tensor:
        """Capture one transformed model call and return its tiled input."""

        observations.append(tiled_args)
        tiled_input = tiled_args["input"]
        if not isinstance(tiled_input, torch.Tensor):
            raise TypeError("Expected tensor input.")
        return tiled_input

    result = TilePredictionAccumulator(plan, diffusion_mode=diffusion_mode).predict(
        args=args,
        x=x,
        evaluate=evaluate,
    )

    if diffusion_mode == "multidiffusion":
        assert torch.equal(result, x)
    else:
        assert torch.allclose(result, x, rtol=1e-6, atol=1e-6)
    assert len(observations) == 1
    transformed = observations[0]
    assert transformed["input"].shape[0] == 4
    assert torch.equal(transformed["timestep"], torch.tensor([0.5, 0.75, 0.5, 0.75]))
    assert transformed["cond_or_uncond"] == [0, 1, 0, 1]
    assert torch.equal(
        transformed["c"]["c_crossattn"],
        torch.tensor([[[1.0]], [[2.0]], [[1.0]], [[2.0]]]),
    )
    options = transformed["c"]["transformer_options"]
    assert options["cond_or_uncond"] == [0, 1, 0, 1]
    assert options["uuids"] == ("positive", "negative", "positive", "negative")
    assert torch.equal(options["sigmas"], transformed["timestep"])
    spatial_layout = options["simple_syrup"]["spatial_batch_layout"]
    assert isinstance(spatial_layout, SpatialBatchLayout)
    assert spatial_layout.input_batch_size == 2
    assert [view.kind for view in spatial_layout.views] == [
        SpatialViewKind.TILE,
        SpatialViewKind.TILE,
    ]
    assert [view.source_x for view in spatial_layout.views] == [0, 4]
    assert "simple_syrup" not in args["c"]["transformer_options"]
    assert args["input"] is x
    assert args["timestep"] is timestep
    assert args["c"]["c_crossattn"] is cross_attention


@pytest.mark.parametrize("layout", ["bchw", "bcdhw"])
def test_accumulator_preserves_distinct_overlap_policies(layout: str) -> None:
    """Fix MultiDiffusion averaging versus Gaussian Mixture weighting."""

    plan = build_tiled_diffusion_plan(6, 4, 4, 4, 2, 2)
    base = torch.arange(6, dtype=torch.float32).reshape((1, 1, 1, 6)).expand(1, 1, 4, 6)
    x = base if layout == "bchw" else base.unsqueeze(2)
    args = {"input": x, "timestep": torch.tensor([1.0]), "c": {}}

    def evaluate(tiled_args: dict[str, Any]) -> torch.Tensor:
        """Return one constant prediction derived from each source tile."""

        tiled_input = tiled_args["input"]
        if not isinstance(tiled_input, torch.Tensor):
            raise TypeError("Expected tensor input.")
        means = tiled_input.mean(dim=(-2, -1), keepdim=True)
        return means.expand_as(tiled_input)

    multidiffusion = TilePredictionAccumulator(
        plan, diffusion_mode="multidiffusion"
    ).predict(args=args, x=x, evaluate=evaluate)
    mixture = TilePredictionAccumulator(
        plan, diffusion_mode="mixture_of_diffusers"
    ).predict(args=args, x=x, evaluate=evaluate)

    assert torch.allclose(multidiffusion[..., 2:4], torch.full_like(x[..., 2:4], 2.5))
    assert not torch.allclose(mixture[..., 2:4], multidiffusion[..., 2:4])
    assert torch.allclose(mixture[..., :2], multidiffusion[..., :2])
    assert torch.allclose(mixture[..., 4:], multidiffusion[..., 4:])


def test_accumulator_caches_gaussian_weights_across_tiles_and_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialize one Gaussian per output layout and tile shape."""

    plan = build_tiled_diffusion_plan(6, 4, 4, 4, 2, 2)
    accumulator = TilePredictionAccumulator(plan, diffusion_mode="mixture_of_diffusers")
    x = torch.ones((1, 1, 4, 6))
    args = {"input": x, "timestep": torch.tensor([1.0]), "c": {}}
    calls: list[tuple[int, int, torch.device, torch.dtype]] = []

    def recorded_weights(
        tile_width: int,
        tile_height: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Record each underlying Gaussian materialization."""

        calls.append((tile_width, tile_height, device, dtype))
        return gaussian_tile_weights(
            tile_width, tile_height, device=device, dtype=dtype
        )

    monkeypatch.setattr(
        "simple_syrup.runtime.tile_prediction_accumulation.gaussian_tile_weights",
        recorded_weights,
    )

    for _ in range(2):
        result = accumulator.predict(
            args=args, x=x, evaluate=lambda tiled: tiled["input"]
        )
        assert torch.equal(result, x)

    assert calls == [(4, 4, x.device, x.dtype)]


def test_semantic_cache_reuses_typed_weights_and_rejects_unknown_tiles() -> None:
    """Keep float32 accumulation authoritative and tile identity explicit."""

    mask = torch.linspace(0.25, 1.0, 16, dtype=torch.float64).reshape((4, 4))
    tile = LatentTile(0, 0, 4, 4, mask)
    cache = SemanticTileWeightCache((tile,))
    output = torch.zeros((2, 3, 1, 4, 4), dtype=torch.float16)

    first = cache.for_output(output)
    second = cache.for_output(output)
    model_weight, accumulation_weight = cache.for_tile(first, tile)

    assert first is second
    assert model_weight.dtype == torch.float16
    assert accumulation_weight.dtype == torch.float32
    assert model_weight.shape == (1, 1, 1, 4, 4)
    assert accumulation_weight.shape == (1, 1, 1, 4, 4)
    with pytest.raises(ValueError, match="unknown tile"):
        cache.for_tile(first, LatentTile(0, 0, 4, 4, mask))


def test_accumulator_rejects_invalid_semantic_mask_shape_at_evaluation() -> None:
    """Fail before blending when a planned semantic mask mismatches its tile."""

    base_plan = build_tiled_diffusion_plan(4, 4, 4, 4, 0, 1)
    invalid_tile = LatentTile(0, 0, 4, 4, torch.ones((3, 4)))
    plan = replace(base_plan, tiles=(invalid_tile,), batches=((invalid_tile,),))
    x = torch.ones((1, 1, 4, 4))

    with pytest.raises(ValueError, match="weight must match its tile dimensions"):
        TilePredictionAccumulator(plan, diffusion_mode="multidiffusion").predict(
            args={"input": x, "timestep": torch.ones((1,)), "c": {}},
            x=x,
            evaluate=lambda tiled: tiled["input"],
        )


def test_accumulator_rejects_unknown_diffusion_mode_during_construction() -> None:
    """Validate the overlap policy before any model evaluation."""

    plan = build_tiled_diffusion_plan(4, 4, 4, 4, 0, 1)

    with pytest.raises(ValueError, match="diffusion_mode must be one of"):
        TilePredictionAccumulator(plan, diffusion_mode="unknown")
