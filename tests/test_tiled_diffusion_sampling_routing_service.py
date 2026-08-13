# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify top-level tiled diffusion route precedence and delegation."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_features import (
    EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    EMPTY_REGIONAL_FEATURE_REQUEST,
)
from simple_syrup.services.tiled_diffusion_sampling_service import (
    TiledDiffusionSamplingService,
)


def test_routing_service_prefers_active_regional_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route prepared regional conditioning before SEGS or batch handling."""

    calls: list[dict[str, Any]] = []
    output = {"samples": torch.ones((1, 4, 4, 4))}

    class RegionalService:
        """Capture the regional route request."""

        def sample(self, **kwargs: Any) -> dict[str, Any]:
            """Record and return the regional result."""

            calls.append(kwargs)
            return output

    class RejectedService:
        """Reject every lower-precedence route."""

        def sample(self, **kwargs: Any) -> dict[str, Any]:
            """Fail if a lower-precedence route is selected."""

            del kwargs
            raise AssertionError("Regional sampling must have precedence.")

    monkeypatch.setattr(
        TiledDiffusionSamplingService,
        "regional_sampling_service_class",
        RegionalService,
    )
    monkeypatch.setattr(
        TiledDiffusionSamplingService,
        "segs_sampling_service_class",
        RejectedService,
    )
    monkeypatch.setattr(
        TiledDiffusionSamplingService,
        "conditioning_batch_service_class",
        RejectedService,
    )
    masks = torch.zeros((2, 4, 4))
    masks[0, :, :2] = 1.0
    masks[1, :, 2:] = 1.0

    result = TiledDiffusionSamplingService().sample(
        **(
            _sample_kwargs()
            | {
                "positive": ConditioningBatch(
                    (
                        _conditioning("global"),
                        _conditioning("left"),
                        _conditioning("right"),
                    )
                ),
                "negative": _conditioning("negative"),
                "region_masks": masks,
                "segs": object(),
            }
        )
    )

    assert result is output
    assert len(calls) == 1
    assert [item[0] for item in calls[0]["positive"]] == [
        "global",
        "left",
        "right",
    ]
    assert calls[0]["region_masks"].shape == (2, 4, 4)
    assert calls[0]["segs"] is not None


def test_routing_service_prefers_segs_over_conditioning_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate a connected SEGS payload before generic batch recursion."""

    calls: list[dict[str, Any]] = []
    output = {"samples": torch.ones((1, 4, 4, 4))}

    class SEGSService:
        """Capture the SEGS route request."""

        def sample(self, **kwargs: Any) -> dict[str, Any]:
            """Record and return the SEGS result."""

            calls.append(kwargs)
            return output

    class RejectedBatchService:
        """Reject generic batch routing."""

        def sample(self, **kwargs: Any) -> dict[str, Any]:
            """Fail if generic batch routing wins precedence."""

            del kwargs
            raise AssertionError("SEGS sampling must have precedence.")

    monkeypatch.setattr(
        TiledDiffusionSamplingService,
        "segs_sampling_service_class",
        SEGSService,
    )
    monkeypatch.setattr(
        TiledDiffusionSamplingService,
        "conditioning_batch_service_class",
        RejectedBatchService,
    )
    segs = object()

    result = TiledDiffusionSamplingService().sample(
        **(
            _sample_kwargs()
            | {
                "positive": ConditioningBatch(("positive",)),
                "segs": segs,
            }
        )
    )

    assert result is output
    assert len(calls) == 1
    assert calls[0]["segs"] is segs
    assert callable(calls[0]["item_sampler"])


def test_routing_service_delegates_conditioning_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate unresolved conditioning batches to their sole service owner."""

    calls: list[dict[str, Any]] = []
    output = {"samples": torch.ones((1, 4, 4, 4))}

    class BatchService:
        """Capture the conditioning batch route request."""

        def sample(self, **kwargs: Any) -> dict[str, Any]:
            """Record and return the batch result."""

            calls.append(kwargs)
            return output

    monkeypatch.setattr(
        TiledDiffusionSamplingService,
        "conditioning_batch_service_class",
        BatchService,
    )
    positive = ConditioningBatch(("positive",))

    result = TiledDiffusionSamplingService().sample(
        **(_sample_kwargs() | {"positive": positive})
    )

    assert result is output
    assert len(calls) == 1
    assert calls[0]["positive"] is positive
    assert callable(calls[0]["item_sampler"])


def test_routing_service_sends_ordinary_request_to_item_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate an ordinary request unchanged to one-item runtime selection."""

    calls: list[dict[str, Any]] = []
    output = {"samples": torch.ones((1, 4, 4, 4))}

    class ItemService:
        """Capture the ordinary item route request."""

        def sample(self, **kwargs: Any) -> dict[str, Any]:
            """Record and return the item result."""

            calls.append(kwargs)
            return output

    monkeypatch.setattr(
        TiledDiffusionSamplingService,
        "item_sampling_service_class",
        ItemService,
    )
    kwargs = _sample_kwargs() | {"region_masks": object()}

    result = TiledDiffusionSamplingService().sample(**kwargs)

    assert result is output
    assert len(calls) == 1
    expected = {
        key: value
        for key, value in kwargs.items()
        if key
        not in {
            "feature_request",
            "region_masks",
            "regional_prompt_weight",
            "region_mask_feather",
            "segs",
        }
    }
    expected["capability_admission"] = EMPTY_REGIONAL_CAPABILITY_ADMISSION
    assert calls[0] == expected


def test_invalid_mode_fails_before_route_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported modes before any downstream owner is constructed."""

    class RejectedPreparationService:
        """Reject route preparation after invalid mode input."""

        def prepare(self, **kwargs: Any) -> object:
            """Fail if validation does not stop routing."""

            del kwargs
            raise AssertionError("Invalid mode must fail before preparation.")

    monkeypatch.setattr(
        TiledDiffusionSamplingService,
        "regional_preparation_service_class",
        RejectedPreparationService,
    )

    with pytest.raises(ValueError, match="diffusion_mode"):
        TiledDiffusionSamplingService().sample(
            **(_sample_kwargs() | {"diffusion_mode": "full_latent"})
        )


def _sample_kwargs() -> dict[str, Any]:
    """Return a complete top-level tiled diffusion request."""

    return {
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
        "feature_request": EMPTY_REGIONAL_FEATURE_REQUEST,
        "segs": None,
        "region_masks": None,
        "regional_prompt_weight": 0.5,
        "region_mask_feather": 0,
    }


def _conditioning(name: str) -> list[list[object]]:
    """Return one structurally valid standard conditioning value."""

    return [[name, {}]]
