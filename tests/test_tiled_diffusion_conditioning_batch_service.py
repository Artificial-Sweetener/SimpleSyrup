"""Verify per-item tiled diffusion conditioning batch execution."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_features import EMPTY_REGIONAL_CAPABILITY_ADMISSION
from simple_syrup.services.tiled_diffusion_conditioning_batch_service import (
    TiledDiffusionConditioningBatchService,
)


def test_batch_service_selects_conditioning_and_slices_latent_metadata() -> None:
    """Select each item's conditioning and aligned batch metadata."""

    calls: list[dict[str, Any]] = []

    def route(**kwargs: Any) -> dict[str, Any]:
        """Capture each routed item and return marked samples."""

        calls.append(kwargs)
        return {
            "samples": torch.full_like(
                kwargs["latent_image"]["samples"],
                float(len(calls)),
            )
        }

    noise_mask = torch.ones((2, 1, 4, 4))
    result = TiledDiffusionConditioningBatchService().sample(
        **(
            _service_kwargs(route=route)
            | {
                "positive": ConditioningBatch(("positive-0", "positive-1")),
                "negative": ConditioningBatch(("negative-last",)),
                "latent_image": {
                    "samples": torch.zeros((2, 4, 4, 4)),
                    "batch_index": [7, 11],
                    "noise_mask": noise_mask,
                    "downscale_ratio_spacial": 2,
                },
            }
        )
    )

    assert [call["positive"] for call in calls] == ["positive-0", "positive-1"]
    assert [call["negative"] for call in calls] == [
        "negative-last",
        "negative-last",
    ]
    assert [call["latent_image"]["batch_index"] for call in calls] == [[7], [11]]
    assert torch.equal(calls[0]["latent_image"]["noise_mask"], noise_mask[0:1])
    assert torch.equal(calls[1]["latent_image"]["noise_mask"], noise_mask[1:2])
    assert "downscale_ratio_spacial" not in result
    assert torch.equal(result["samples"][0], torch.full((4, 4, 4), 1.0))
    assert torch.equal(result["samples"][1], torch.full((4, 4, 4), 2.0))


def test_batch_service_forwards_sampling_arguments_unchanged() -> None:
    """Preserve every non-batch request value at the recursive route boundary."""

    calls: list[dict[str, Any]] = []

    def route(**kwargs: Any) -> dict[str, Any]:
        """Capture the complete item route request."""

        calls.append(kwargs)
        return {"samples": kwargs["latent_image"]["samples"]}

    kwargs = _service_kwargs(route=route)
    TiledDiffusionConditioningBatchService().sample(**kwargs)

    expected_values = {
        key: value
        for key, value in kwargs.items()
        if key not in {"item_sampler", "positive", "latent_image"}
    }
    assert len(calls) == 1
    assert {
        key: value for key, value in calls[0].items() if key != "latent_image"
    } == expected_values | {"positive": "positive"}
    assert torch.equal(
        calls[0]["latent_image"]["samples"],
        kwargs["latent_image"]["samples"],
    )


def test_batch_service_rejects_non_tensor_latent_samples() -> None:
    """Fail before routing when the latent payload is malformed."""

    with pytest.raises(TypeError, match="latent samples must be a torch.Tensor"):
        TiledDiffusionConditioningBatchService().sample(
            **(
                _service_kwargs(route=_unchanged_route)
                | {"latent_image": {"samples": object()}}
            )
        )


def test_batch_service_rejects_non_tensor_route_output() -> None:
    """Reject malformed route output before recombining a latent batch."""

    def invalid_output(**kwargs: Any) -> dict[str, Any]:
        """Return an invalid samples value."""

        del kwargs
        return {"samples": object()}

    with pytest.raises(TypeError, match="output samples must be a torch.Tensor"):
        TiledDiffusionConditioningBatchService().sample(
            **_service_kwargs(route=invalid_output)
        )


def _unchanged_route(**kwargs: Any) -> dict[str, Any]:
    """Return one routed latent item unchanged."""

    return {"samples": kwargs["latent_image"]["samples"]}


def _service_kwargs(*, route: Any) -> dict[str, Any]:
    """Return a complete conditioning batch request."""

    return {
        "item_sampler": route,
        "diffusion_mode": "multidiffusion",
        "model": "model",
        "seed": 123,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": ConditioningBatch(("positive",)),
        "negative": "negative",
        "latent_image": {"samples": torch.zeros((1, 4, 4, 4))},
        "denoise": 0.8,
        "latent_tile_width": 128,
        "latent_tile_height": 80,
        "latent_tile_overlap": 24,
        "latent_tile_batch_size": 3,
        "preview_context": None,
        "differential_diffusion": True,
        "capability_admission": EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    }
