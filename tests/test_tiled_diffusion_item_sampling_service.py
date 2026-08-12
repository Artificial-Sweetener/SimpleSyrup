"""Verify one-item tiled diffusion runtime selection and forwarding."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.regional_features import EMPTY_REGIONAL_CAPABILITY_ADMISSION
from simple_syrup.services.tiled_diffusion_item_sampling_service import (
    TiledDiffusionItemSamplingService,
)


@pytest.mark.parametrize(
    ("diffusion_mode", "selected", "rejected"),
    [
        ("multidiffusion", "multidiffusion", "mixture"),
        ("mixture_of_diffusers", "mixture", "multidiffusion"),
    ],
)
def test_item_service_selects_exactly_one_runtime(
    diffusion_mode: str,
    selected: str,
    rejected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route each admitted mode to only its authoritative runtime."""

    calls: list[str] = []
    output = {"samples": torch.ones((1, 4, 4, 4))}

    def multidiffusion(**kwargs: Any) -> dict[str, Any]:
        """Record a MultiDiffusion invocation."""

        del kwargs
        calls.append("multidiffusion")
        return output

    def mixture(**kwargs: Any) -> dict[str, Any]:
        """Record a Mixture-of-Diffusers invocation."""

        del kwargs
        calls.append("mixture")
        return output

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_item_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        multidiffusion,
    )
    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_item_sampling_service."
        "mixture_of_diffusers_sampling.sample_mixture_of_diffusers",
        mixture,
    )

    result = TiledDiffusionItemSamplingService().sample(
        **_item_kwargs(diffusion_mode=diffusion_mode)
    )

    assert result is output
    assert calls == [selected]
    assert rejected not in calls


def test_item_service_forwards_every_sampling_argument_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the complete request and explicit absent-plan value."""

    calls: dict[str, Any] = {}
    output = {"samples": torch.ones((1, 4, 4, 4))}

    def record(**kwargs: Any) -> dict[str, Any]:
        """Capture the selected runtime request."""

        calls.update(kwargs)
        return output

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_item_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        record,
    )
    kwargs = _item_kwargs(diffusion_mode="multidiffusion")

    result = TiledDiffusionItemSamplingService().sample(**kwargs)

    assert result is output
    assert calls == {
        key: value for key, value in kwargs.items() if key != "diffusion_mode"
    }


def test_item_service_forwards_differential_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the differential denoise-mask composition request."""

    calls: dict[str, Any] = {}

    def record(**kwargs: Any) -> dict[str, Any]:
        """Capture the selected runtime request."""

        calls.update(kwargs)
        return {"samples": kwargs["latent_image"]["samples"]}

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_item_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        record,
    )
    TiledDiffusionItemSamplingService().sample(
        **(
            _item_kwargs(diffusion_mode="multidiffusion")
            | {"differential_diffusion": True}
        )
    )

    assert calls["differential_diffusion"] is True


def _item_kwargs(*, diffusion_mode: str) -> dict[str, Any]:
    """Return a complete one-item runtime request."""

    return {
        "diffusion_mode": diffusion_mode,
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
        "preview_context": object(),
        "differential_diffusion": False,
        "capability_admission": EMPTY_REGIONAL_CAPABILITY_ADMISSION,
        "tiled_plan": None,
    }
