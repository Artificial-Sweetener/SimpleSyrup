# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for AUTOMATIC1111-derived sampler functions."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

import pytest
import torch

from simple_syrup.runtime import a1111_sampling


class FakeModel:
    """Provide a deterministic denoiser for sampler-loop tests."""

    def __init__(self) -> None:
        """Create a fake model with call recording."""

        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return a deterministic denoised tensor and record call arguments."""

        self.calls.append({"x": x.clone(), "sigma": sigma.clone(), "kwargs": kwargs})
        return x * 0.5


def reference_euler_ancestral(
    model: Callable[..., torch.Tensor],
    x: torch.Tensor,
    sigmas: torch.Tensor,
    noise_sampler: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Calculate the A1111/k-diffusion Euler ancestral reference loop."""

    s_in = x.new_ones([x.shape[0]])
    for index in range(len(sigmas) - 1):
        denoised = model(x, sigmas[index] * s_in)
        sigma_down, sigma_up = reference_ancestral_step(
            sigmas[index],
            sigmas[index + 1],
        )
        derivative = (x - denoised) / sigmas[index]
        x = x + derivative * (sigma_down - sigmas[index])
        if sigmas[index + 1] > 0:
            x = x + noise_sampler(sigmas[index], sigmas[index + 1]) * sigma_up
    return x


def reference_ancestral_step(
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate k-diffusion's default eta=1 ancestral step."""

    sigma_up = torch.minimum(
        sigma_to,
        (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2).sqrt(),
    )
    sigma_down = (sigma_to**2 - sigma_up**2).sqrt()
    return sigma_down, sigma_up


def test_euler_a_a1111_matches_reference_loop_with_deterministic_noise() -> None:
    """The local sampler follows the A1111/k-diffusion Euler ancestral loop."""

    x = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    noise_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def noise_sampler(
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """Return deterministic ancestral noise and record call arguments."""

        noise_calls.append((sigma.clone(), sigma_next.clone()))
        return torch.full_like(x, 0.25)

    expected = reference_euler_ancestral(FakeModel(), x.clone(), sigmas, noise_sampler)
    noise_calls.clear()

    actual = a1111_sampling.sample_euler_ancestral_a1111(
        FakeModel(),
        x.clone(),
        sigmas,
        noise_sampler=noise_sampler,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert len(noise_calls) == 1
    assert torch.equal(noise_calls[0][0], sigmas[0])
    assert torch.equal(noise_calls[0][1], sigmas[1])


def test_euler_a_a1111_callback_and_extra_args() -> None:
    """The sampler forwards extra args and emits k-diffusion callback payloads."""

    model = FakeModel()
    callback_payloads: list[dict[str, object]] = []
    x = torch.ones((1, 2), dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)

    a1111_sampling.sample_euler_ancestral_a1111(
        model,
        x,
        sigmas,
        extra_args={"seed": 123, "model_options": {"kept": True}},
        callback=callback_payloads.append,
        noise_sampler=lambda _sigma, _sigma_next: torch.zeros_like(x),
    )

    assert model.calls[0]["kwargs"] == {
        "seed": 123,
        "model_options": {"kept": True},
    }
    assert len(callback_payloads) == 1
    payload = callback_payloads[0]
    assert payload["i"] == 0
    assert isinstance(payload["sigma"], torch.Tensor)
    assert isinstance(payload["sigma_hat"], torch.Tensor)
    assert torch.equal(payload["sigma"], sigmas[0])
    assert torch.equal(payload["sigma_hat"], sigmas[0])
    assert isinstance(payload["denoised"], torch.Tensor)


def test_euler_a_a1111_uses_arithmetic_path_for_final_zero_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final zero transition still calls to_d instead of Comfy's shortcut."""

    calls: list[torch.Tensor] = []

    def fake_to_d(
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoised: torch.Tensor,
    ) -> torch.Tensor:
        """Record derivative conversion for the final zero step."""

        del sigma, denoised
        calls.append(x.clone())
        return torch.ones_like(x)

    monkeypatch.setattr(a1111_sampling, "_to_d", fake_to_d)

    a1111_sampling.sample_euler_ancestral_a1111(
        FakeModel(),
        torch.ones((1, 2), dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        noise_sampler=lambda _sigma, _sigma_next: torch.zeros((1, 2)),
    )

    assert len(calls) == 1


def test_euler_a_a1111_default_noise_sampler_uses_comfy_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default noise delegates to ComfyUI's deterministic seed-aware helper."""

    comfy_sampling = import_module("comfy.k_diffusion.sampling")
    calls: list[object] = []

    def fake_default_noise_sampler(
        x: torch.Tensor,
        seed: object = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Record seed delegation and return deterministic noise."""

        calls.append(seed)
        return lambda _sigma, _sigma_next: torch.zeros_like(x)

    monkeypatch.setattr(
        comfy_sampling,
        "default_noise_sampler",
        fake_default_noise_sampler,
    )

    a1111_sampling.sample_euler_ancestral_a1111(
        FakeModel(),
        torch.ones((1, 2), dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        extra_args={"seed": 456},
    )

    assert calls == [456]
