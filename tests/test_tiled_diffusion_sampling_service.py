# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shared tiled diffusion sampling mode dispatch."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.services.tiled_diffusion_sampling_service import (
    TiledDiffusionSamplingService,
)


def test_multidiffusion_mode_routes_to_multidiffusion_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MultiDiffusion mode calls only the MultiDiffusion runtime."""

    calls: dict[str, dict[str, Any]] = {}
    output: dict[str, Any] = {"samples": torch.ones((1, 4, 4, 4))}

    def fake_multidiffusion(**kwargs: Any) -> dict[str, Any]:
        """Record MultiDiffusion runtime arguments."""

        calls["multidiffusion"] = kwargs
        return output

    def fake_mixture(**kwargs: Any) -> dict[str, Any]:
        """Fail if Mixture runtime is selected."""

        calls["mixture"] = kwargs
        raise AssertionError("Mixture runtime should not be called.")

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        fake_multidiffusion,
    )
    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "mixture_of_diffusers_sampling.sample_mixture_of_diffusers",
        fake_mixture,
    )

    result = TiledDiffusionSamplingService().sample(
        **_sample_kwargs(diffusion_mode="multidiffusion")
    )

    assert result is output
    assert "multidiffusion" in calls
    assert "mixture" not in calls


def test_mixture_mode_routes_to_mixture_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixture of Diffusers mode calls only the Mixture runtime."""

    calls: dict[str, dict[str, Any]] = {}
    output: dict[str, Any] = {"samples": torch.ones((1, 4, 4, 4))}

    def fake_multidiffusion(**kwargs: Any) -> dict[str, Any]:
        """Fail if MultiDiffusion runtime is selected."""

        calls["multidiffusion"] = kwargs
        raise AssertionError("MultiDiffusion runtime should not be called.")

    def fake_mixture(**kwargs: Any) -> dict[str, Any]:
        """Record Mixture runtime arguments."""

        calls["mixture"] = kwargs
        return output

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        fake_multidiffusion,
    )
    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "mixture_of_diffusers_sampling.sample_mixture_of_diffusers",
        fake_mixture,
    )

    result = TiledDiffusionSamplingService().sample(
        **_sample_kwargs(diffusion_mode="mixture_of_diffusers")
    )

    assert result is output
    assert "mixture" in calls
    assert "multidiffusion" not in calls


def test_service_forwards_sampling_arguments_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher preserves every sampler argument at the runtime boundary."""

    calls: dict[str, Any] = {}
    output: dict[str, Any] = {"samples": torch.ones((1, 4, 4, 4))}
    preview_context = object()

    def fake_multidiffusion(**kwargs: Any) -> dict[str, Any]:
        """Record forwarded arguments."""

        calls.update(kwargs)
        return output

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        fake_multidiffusion,
    )

    kwargs = _sample_kwargs(
        diffusion_mode="multidiffusion",
        preview_context=preview_context,
    )
    result = TiledDiffusionSamplingService().sample(**kwargs)

    assert result is output
    assert calls == {
        key: value for key, value in kwargs.items() if key != "diffusion_mode"
    }


def test_service_forwards_differential_diffusion_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher preserves differential-denoise-mask composition requests."""

    calls: dict[str, Any] = {}

    def fake_multidiffusion(**kwargs: Any) -> dict[str, Any]:
        """Record forwarded arguments."""

        calls.update(kwargs)
        return {"samples": torch.ones((1, 4, 4, 4))}

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        fake_multidiffusion,
    )

    TiledDiffusionSamplingService().sample(
        **(
            _sample_kwargs(diffusion_mode="multidiffusion")
            | {"differential_diffusion": True}
        )
    )

    assert calls["differential_diffusion"] is True


def test_service_selects_conditioning_batch_per_latent_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch conditioning is selected before tiled runtime dispatch."""

    calls: list[dict[str, Any]] = []

    def fake_multidiffusion(**kwargs: Any) -> dict[str, Any]:
        """Record per-item runtime arguments and return marked samples."""

        calls.append(kwargs)
        return {
            "samples": torch.full_like(
                kwargs["latent_image"]["samples"],
                float(len(calls)),
            )
        }

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        fake_multidiffusion,
    )

    latent_samples = torch.zeros((2, 4, 4, 4))
    noise_mask = torch.ones((2, 1, 4, 4))
    result = TiledDiffusionSamplingService().sample(
        **(
            _sample_kwargs(diffusion_mode="multidiffusion")
            | {
                "positive": ConditioningBatch(("positive-0", "positive-1")),
                "negative": ConditioningBatch(("negative-last",)),
                "latent_image": {
                    "samples": latent_samples,
                    "batch_index": [7, 11],
                    "noise_mask": noise_mask,
                    "downscale_ratio_spacial": 2,
                },
            }
        )
    )

    assert len(calls) == 2
    assert calls[0]["positive"] == "positive-0"
    assert calls[1]["positive"] == "positive-1"
    assert calls[0]["negative"] == "negative-last"
    assert calls[1]["negative"] == "negative-last"
    assert calls[0]["latent_image"]["batch_index"] == [7]
    assert calls[1]["latent_image"]["batch_index"] == [11]
    assert torch.equal(calls[0]["latent_image"]["noise_mask"], noise_mask[0:1])
    assert torch.equal(calls[1]["latent_image"]["noise_mask"], noise_mask[1:2])
    assert "downscale_ratio_spacial" not in result
    assert result["samples"].shape == latent_samples.shape
    assert torch.equal(result["samples"][0], torch.full((4, 4, 4), 1.0))
    assert torch.equal(result["samples"][1], torch.full((4, 4, 4), 2.0))


def test_invalid_mode_fails_before_runtime_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported modes are rejected before any runtime sampler is called."""

    def fail_runtime(**kwargs: Any) -> dict[str, Any]:
        """Fail if validation does not stop dispatch."""

        del kwargs
        raise AssertionError("Runtime should not be called.")

    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "multidiffusion_sampling.sample_multidiffusion",
        fail_runtime,
    )
    monkeypatch.setattr(
        "simple_syrup.services.tiled_diffusion_sampling_service."
        "mixture_of_diffusers_sampling.sample_mixture_of_diffusers",
        fail_runtime,
    )

    with pytest.raises(ValueError, match="diffusion_mode"):
        TiledDiffusionSamplingService().sample(
            **_sample_kwargs(diffusion_mode="full_latent")
        )


def _sample_kwargs(
    *,
    diffusion_mode: str,
    preview_context: object | None = None,
) -> dict[str, Any]:
    """Return valid tiled diffusion sample arguments."""

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
        "preview_context": preview_context,
        "differential_diffusion": False,
    }
