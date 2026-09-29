# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify RES4LYF selection survives contextual and tiled model composition."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from simple_syrup.domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from simple_syrup.runtime import (
    contextual_diffusion_sampling,
    mixture_of_diffusers_sampling,
    multidiffusion_sampling,
    sampling_schedulers,
)
from simple_syrup.runtime.sampling_noise import prepare_sampling_noise


class _Model:
    """Provide the patcher and sigma bounds used at each composed boundary."""

    def __init__(
        self, model_options: dict[str, Any] | None = None, parent: _Model | None = None
    ) -> None:
        """Create a CPU model with copyable wrapper options."""

        self.load_device = torch.device("cpu")
        self.model_options = {} if model_options is None else model_options
        self.sampling = SimpleNamespace(sigma_max=1.0, sigma_min=0.0)
        self.parent = parent

    def clone(self) -> _Model:
        """Preserve model sampling and copy wrapper options."""

        cloned = _Model(self.model_options.copy(), parent=self)
        cloned.sampling = self.sampling
        return cloned

    def set_model_unet_function_wrapper(self, wrapper: object) -> None:
        """Record the contextual or tiled prediction wrapper."""

        self.model_options["model_function_wrapper"] = wrapper

    def get_model_object(self, name: str) -> object:
        """Return the same sigma bounds before and after model cloning."""

        assert name == "model_sampling"
        return self.sampling


@pytest.mark.parametrize(
    "mode",
    ["contextual", "multidiffusion", "mixture_of_diffusers"],
)
def test_res4lyf_sampler_reaches_composed_sampling_boundary(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Pass the real solver, RES noise, and wrapped model into ComfyUI."""

    model = _Model()
    latent_samples = torch.zeros((1, 4, 16, 32))
    latent = {"samples": latent_samples}
    runtime = {
        "contextual": contextual_diffusion_sampling,
        "multidiffusion": multidiffusion_sampling,
        "mixture_of_diffusers": mixture_of_diffusers_sampling,
    }[mode]
    comfy_sample = runtime._comfy_sample()
    preview = runtime._latent_preview()
    sigma_calls: list[dict[str, object]] = []
    sampled: list[dict[str, object]] = []

    def calculate_sigmas(**kwargs: object) -> torch.Tensor:
        """Capture the RES4LYF scheduler and return a short finite schedule."""

        sigma_calls.append(kwargs)
        return torch.tensor([1.0, 0.5, 0.0])

    def sample_custom(
        sampling_model: _Model,
        noise: torch.Tensor,
        cfg: float,
        sampler: object,
        sigmas: torch.Tensor,
        positive: object,
        negative: object,
        latent_image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Inspect the arguments at ComfyUI's composed sampling boundary."""

        del cfg, sigmas, positive, negative, kwargs
        sampled.append({"model": sampling_model, "noise": noise, "sampler": sampler})
        return latent_image + 1

    monkeypatch.setattr(sampling_schedulers, "calculate_sigmas", calculate_sigmas)
    monkeypatch.setattr(
        comfy_sample,
        "fix_empty_latent_channels",
        lambda _model, samples, _ratio: samples,
    )
    monkeypatch.setattr(
        comfy_sample,
        "prepare_noise",
        lambda *_args: pytest.fail("RES4LYF must generate its own initial noise"),
    )
    monkeypatch.setattr(comfy_sample, "sample_custom", sample_custom)
    monkeypatch.setattr(preview, "prepare_callback", lambda _model, _steps: None)

    controls = ContextualDiffusionControls(16, 0, 2, 1.0, 1, 0.5)
    plan = build_contextual_diffusion_plan(
        latent_width=32,
        latent_height=16,
        controls=controls,
        segs=None,
    )
    common = {
        "model": model,
        "seed": 42,
        "steps": 2,
        "cfg": 1.0,
        "sampler_name": "exponential/ddim",
        "scheduler": "bong_tangent",
        "positive": [],
        "negative": [],
        "latent_image": latent,
        "denoise": 1.0,
    }
    if mode == "contextual":
        sample_runtime: Callable[..., dict[str, Any]] = (
            contextual_diffusion_sampling.sample_contextual_diffusion
        )
        extra = {"diffusion_mode": "multidiffusion", "controls": controls, "plan": plan}
    else:
        sample_runtime = (
            multidiffusion_sampling.sample_multidiffusion
            if mode == "multidiffusion"
            else mixture_of_diffusers_sampling.sample_mixture_of_diffusers
        )
        extra = {
            "latent_tile_width": 16,
            "latent_tile_height": 16,
            "latent_tile_overlap": 0,
            "latent_tile_batch_size": 2,
        }
    output = sample_runtime(**common, **extra)

    assert torch.equal(output["samples"], latent_samples + 1)
    assert sigma_calls[0]["scheduler_name"] == "bong_tangent"
    assert sigma_calls[0]["sampler_name"] == "exponential/ddim"
    assert len(sampled) == 1
    derived_model = sampled[0]["model"]
    assert isinstance(derived_model, _Model)
    assert derived_model is not model
    assert callable(derived_model.model_options["model_function_wrapper"])
    sampler = sampled[0]["sampler"]
    assert vars(sampler)["extra_options"]["rk_type"] == "ddim"
    expected_noise = prepare_sampling_noise(
        comfy_sample=comfy_sample,
        sampler_name="exponential/ddim",
        samples=latent_samples,
        seed=42,
        batch_indices=None,
        model=derived_model,
    )
    torch.testing.assert_close(sampled[0]["noise"], expected_noise, rtol=0, atol=0)
