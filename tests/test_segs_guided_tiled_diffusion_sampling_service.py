"""Verify SEGS-guided tiled diffusion batch planning and execution."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_features import EMPTY_REGIONAL_CAPABILITY_ADMISSION
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.services.segs_guided_tiled_diffusion_sampling_service import (
    SEGSGuidedTiledDiffusionSamplingService,
)


def test_segs_service_builds_and_forwards_a_guided_plan() -> None:
    """Replace the regular grid with a semantic plan for one latent item."""

    calls: list[dict[str, Any]] = []

    def item_sampler(**kwargs: Any) -> dict[str, Any]:
        """Capture one planned item request."""

        calls.append(kwargs)
        return {"samples": kwargs["latent_image"]["samples"]}

    result = SEGSGuidedTiledDiffusionSamplingService().sample(
        **(_service_kwargs(item_sampler=item_sampler) | {"segs": _segs(4, 4)})
    )

    assert result["samples"].shape == (1, 4, 4, 4)
    assert len(calls) == 1
    plan = calls[0]["tiled_plan"]
    assert plan.latent_width == 4
    assert plan.latent_height == 4
    assert all(tile.weight_mask is not None for tile in plan.tiles)


def test_segs_service_selects_conditioning_for_each_latent_item() -> None:
    """Preserve shared SEGS while selecting each item's conditioning."""

    calls: list[dict[str, Any]] = []

    def item_sampler(**kwargs: Any) -> dict[str, Any]:
        """Capture per-item conditioning and return marked samples."""

        calls.append(kwargs)
        return {
            "samples": torch.full_like(
                kwargs["latent_image"]["samples"],
                float(len(calls)),
            )
        }

    latent = {
        "samples": torch.zeros((2, 4, 4, 4)),
        "batch_index": [7, 11],
        "noise_mask": torch.ones((2, 1, 4, 4)),
        "downscale_ratio_spacial": 2,
    }
    result = SEGSGuidedTiledDiffusionSamplingService().sample(
        **(
            _service_kwargs(item_sampler=item_sampler)
            | {
                "positive": ConditioningBatch(("positive-0", "positive-1")),
                "negative": ConditioningBatch(("negative-last",)),
                "latent_image": latent,
                "segs": _segs(4, 4),
            }
        )
    )

    assert [call["positive"] for call in calls] == ["positive-0", "positive-1"]
    assert [call["negative"] for call in calls] == [
        "negative-last",
        "negative-last",
    ]
    assert [call["latent_image"]["batch_index"] for call in calls] == [[7], [11]]
    assert all(call["tiled_plan"] is not None for call in calls)
    assert "downscale_ratio_spacial" not in result
    assert torch.equal(result["samples"][0], torch.full((4, 4, 4), 1.0))
    assert torch.equal(result["samples"][1], torch.full((4, 4, 4), 2.0))


def test_segs_service_rejects_batch_alignment_mismatch() -> None:
    """Reject SEGS groups that cannot align with the latent batch."""

    with pytest.raises(ValueError, match="one SEGS payload or one per latent"):
        SEGSGuidedTiledDiffusionSamplingService().sample(
            **(
                _service_kwargs(item_sampler=_unchanged_item_sampler)
                | {
                    "latent_image": {"samples": torch.zeros((3, 4, 4, 4))},
                    "segs": (_segs(4, 4), _segs(4, 4)),
                }
            )
        )


def test_segs_service_rejects_non_tensor_latent_samples() -> None:
    """Fail before planning when the latent payload is malformed."""

    with pytest.raises(TypeError, match="latent samples must be a torch.Tensor"):
        SEGSGuidedTiledDiffusionSamplingService().sample(
            **(
                _service_kwargs(item_sampler=_unchanged_item_sampler)
                | {"latent_image": {"samples": object()}, "segs": _segs(4, 4)}
            )
        )


def test_segs_service_rejects_non_tensor_sampler_output() -> None:
    """Reject malformed item sampler output before combining a batch."""

    def invalid_output(**kwargs: Any) -> dict[str, Any]:
        """Return an invalid samples value."""

        del kwargs
        return {"samples": object()}

    with pytest.raises(TypeError, match="output samples must be a torch.Tensor"):
        SEGSGuidedTiledDiffusionSamplingService().sample(
            **(_service_kwargs(item_sampler=invalid_output) | {"segs": _segs(4, 4)})
        )


def _unchanged_item_sampler(**kwargs: Any) -> dict[str, Any]:
    """Return the supplied item's samples unchanged."""

    return {"samples": kwargs["latent_image"]["samples"]}


def _service_kwargs(*, item_sampler: Any) -> dict[str, Any]:
    """Return a complete SEGS-guided sampling request."""

    return {
        "item_sampler": item_sampler,
        "segs": _segs(4, 4),
        "diffusion_mode": "multidiffusion",
        "model": "model",
        "seed": 123,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": "positive",
        "negative": "negative",
        "latent_image": {"samples": torch.zeros((1, 4, 4, 4))},
        "denoise": 0.8,
        "latent_tile_width": 128,
        "latent_tile_height": 80,
        "latent_tile_overlap": 24,
        "latent_tile_batch_size": 3,
        "preview_context": None,
        "differential_diffusion": False,
        "capability_admission": EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    }


def _segs(height: int, width: int) -> tuple[tuple[int, int], tuple[Segment, ...]]:
    """Return one full-image SEGS payload."""

    region = CropRegion(0, 0, width, height)
    segment = Segment(
        cropped_image=None,
        cropped_mask=torch.ones((height, width), dtype=torch.float32),
        confidence=1.0,
        crop_region=region,
        bbox=BoundingBox(0, 0, width, height),
        label="region",
    )
    return ((height, width), (segment,))
