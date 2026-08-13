# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify regional tiled diffusion planning and batch execution."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.regional_features import (
    RegionalCapabilityAdmission,
    RegionalFeature,
    RegionalFeatureRequest,
)
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.services.regional_tiled_diffusion_sampling_service import (
    RegionalTiledDiffusionSamplingService,
)


def test_regional_service_builds_and_forwards_region_constrained_plan() -> None:
    """Build one plan whose tiles remain constrained to authored regions."""

    calls: list[dict[str, Any]] = []

    def item_sampler(**kwargs: Any) -> dict[str, Any]:
        """Capture the regional item request."""

        calls.append(kwargs)
        return {"samples": kwargs["latent_image"]["samples"]}

    result = RegionalTiledDiffusionSamplingService().sample(
        **_service_kwargs(item_sampler=item_sampler)
    )

    assert result["samples"].shape == (1, 4, 4, 4)
    assert len(calls) == 1
    assert calls[0]["capability_admission"].supports(
        RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING
    )
    assert len(calls[0]["tiled_plan"].tiles) == 2
    assert all(tile.weight_mask is not None for tile in calls[0]["tiled_plan"].tiles)


def test_regional_service_reuses_one_segs_payload_across_latent_batch() -> None:
    """Apply a shared semantic guide while preserving each latent item."""

    calls: list[dict[str, Any]] = []

    def item_sampler(**kwargs: Any) -> dict[str, Any]:
        """Capture each regional item request."""

        calls.append(kwargs)
        return {"samples": kwargs["latent_image"]["samples"]}

    RegionalTiledDiffusionSamplingService().sample(
        **(
            _service_kwargs(item_sampler=item_sampler)
            | {
                "latent_image": {"samples": torch.zeros((2, 4, 4, 4))},
                "segs": _segs(4, 4),
            }
        )
    )

    assert len(calls) == 2
    assert all(call["tiled_plan"] is not None for call in calls)


def test_regional_service_rejects_segs_batch_alignment_mismatch() -> None:
    """Reject semantic guides that cannot align with the latent batch."""

    with pytest.raises(ValueError, match="one SEGS payload or one per latent"):
        RegionalTiledDiffusionSamplingService().sample(
            **(
                _service_kwargs(item_sampler=_unchanged_item_sampler)
                | {
                    "latent_image": {"samples": torch.zeros((3, 4, 4, 4))},
                    "segs": (_segs(4, 4), _segs(4, 4)),
                }
            )
        )


def test_regional_service_rejects_malformed_input_and_output() -> None:
    """Fail closed at both latent tensor boundaries."""

    with pytest.raises(TypeError, match="latent samples must be a torch.Tensor"):
        RegionalTiledDiffusionSamplingService().sample(
            **(
                _service_kwargs(item_sampler=_unchanged_item_sampler)
                | {"latent_image": {"samples": object()}}
            )
        )

    def invalid_output(**kwargs: Any) -> dict[str, Any]:
        """Return an invalid samples value."""

        del kwargs
        return {"samples": object()}

    with pytest.raises(TypeError, match="output samples must be a torch.Tensor"):
        RegionalTiledDiffusionSamplingService().sample(
            **_service_kwargs(item_sampler=invalid_output)
        )


def _unchanged_item_sampler(**kwargs: Any) -> dict[str, Any]:
    """Return one latent item unchanged."""

    return {"samples": kwargs["latent_image"]["samples"]}


def _service_kwargs(*, item_sampler: Any) -> dict[str, Any]:
    """Return a complete regional sampling request."""

    masks = torch.zeros((2, 4, 4))
    masks[0, :, :2] = 1.0
    masks[1, :, 2:] = 1.0
    return {
        "item_sampler": item_sampler,
        "region_masks": masks,
        "segs": None,
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
        "capability_admission": _full_context_admission(),
    }


def _full_context_admission() -> RegionalCapabilityAdmission:
    """Return successful full-context admission for direct service tests."""

    request = RegionalFeatureRequest(
        frozenset({RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING})
    )
    return RegionalCapabilityAdmission(request, request.features, None)


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
