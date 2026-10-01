# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify real Comfy scaling with isolated external guider/model execution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from comfy.model_sampling import CONST, EPS

from simple_syrup.domain.noise_inversion import NoiseInversionOptions
from simple_syrup.runtime import guided_sampling, noise_inversion


def _flow() -> Any:
    """Use the actual Comfy flow scaling object at the dynamic host boundary."""
    sampling = CONST()
    sampling.sigma_max = 1.0
    sampling.noise_scale = 1.0
    return sampling


def _epsilon() -> Any:
    """Use the actual Comfy EPS scaling object at the dynamic host boundary."""
    sampling = EPS()
    sampling.sigma_max = 10.0
    return sampling


def _model(sampling: Any) -> Any:
    """Represent only the dynamic Comfy patcher fields needed by inversion."""
    return SimpleNamespace(
        load_device=torch.device("cpu"),
        get_model_object=lambda name: sampling,
        model=SimpleNamespace(process_latent_in=lambda x: x),
    )


@pytest.fixture
def guider_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace neural execution but retain the actual Comfy noise scaling contract."""
    calls: list[dict[str, Any]] = []
    ticks = iter(range(100))
    monkeypatch.setattr(noise_inversion, "_clock", lambda device: float(next(ticks)))

    def sample_boundary(**kwargs: Any) -> torch.Tensor:
        """Drive the registered custom sampler with a closed-form denoiser."""
        calls.append(kwargs)
        sigmas = kwargs["sigmas"]
        latent = kwargs["latent_image"]
        sampling = kwargs["model"].get_model_object("model_sampling")
        state = sampling.noise_scaling(
            sigmas[0].double(),
            kwargs["noise"].double(),
            latent.double(),
            max_denoise=False,
        )

        def denoiser(
            x: torch.Tensor, sigma: torch.Tensor, **extra: Any
        ) -> torch.Tensor:
            """Expose constant velocity two through Comfy's denoised prediction."""
            return x - 2 * sigma.reshape((-1,) + (1,) * (x.ndim - 1))

        return cast(
            torch.Tensor,
            kwargs["sampler"].sampler_function(
                denoiser, state, sigmas, extra_args={}, callback=None, disable=True
            ),
        )

    monkeypatch.setattr(
        guided_sampling, "sample_with_optional_negative", sample_boundary
    )
    return calls


@pytest.mark.parametrize("negative", [None, [[torch.ones((1, 1, 1)), {}]]])
@pytest.mark.parametrize("singleton_depth", [False, True])
def test_default_recipe_executes_coarse_and_full_stages_with_original_conditioning(
    guider_calls: list[dict[str, Any]], negative: Any, singleton_depth: bool
) -> None:
    """Keep the accepted two-plus-one recipe and positive-only path intact."""
    shape = (1, 4, 1, 8, 12) if singleton_depth else (1, 4, 8, 12)
    latent = torch.full(shape, 0.7)
    positive = [[torch.ones((1, 1, 1)), {}]]
    model = _model(_flow())
    factory_shapes: list[tuple[int, ...]] = []

    def factory(stage_latent: torch.Tensor) -> Any:
        """Record the independently planned stage dimensions."""
        factory_shapes.append(tuple(stage_latent.shape))
        return model

    result = noise_inversion.invert_sampling_noise(
        model=model,
        latent=latent,
        forward_sigmas=torch.tensor([0.5, 0.25, 0]),
        positive=positive,
        negative=negative,
        cfg=3,
        seed=123,
        options=NoiseInversionOptions(),
        model_factory=factory,
    )
    assert [phase.name for phase in result.stages] == ["coarse", "full_finish"]
    assert [phase.steps for phase in result.stages] == [2, 1]
    assert [phase.evaluations for phase in result.stages] == [2, 1]
    assert factory_shapes == [shape[:-2] + (4, 6), shape]
    assert len(guider_calls) == 2
    for call in guider_calls:
        assert call["positive"] is positive
        assert call["negative"] is negative
        assert call["cfg"] == 3 and call["seed"] == 123
    assert float(guider_calls[0]["sigmas"][-1]) == 0.375
    assert float(guider_calls[1]["sigmas"][0]) == 0.375
    assert float(guider_calls[1]["sigmas"][-1]) == 0.5
    expected_endpoint = latent * (1 - 0.0001) + 2 * (0.5 - 0.0001)
    reconstructed = _flow().noise_scaling(torch.tensor(0.5), result.noise, latent)
    assert torch.allclose(reconstructed, expected_endpoint, atol=1e-6)
    assert result.reconstruction_max_error <= 1e-6
    assert result.seconds == 5
    assert torch.equal(latent, torch.full(shape, 0.7))


@pytest.mark.parametrize("sampling", [_flow(), _epsilon()])
def test_full_size_euler_and_heun_recreate_endpoint_for_both_noise_scalings(
    guider_calls: list[dict[str, Any]], sampling: Any
) -> None:
    """Recover actual noise, not a flow-only formula wrongly applied to EPS models."""
    latent = torch.full((1, 2, 7, 9), 0.7)
    for method in ("euler", "heun"):
        result = noise_inversion.invert_sampling_noise(
            model=_model(sampling),
            latent=latent,
            forward_sigmas=torch.tensor([0.5, 0]),
            positive=[],
            negative=None,
            cfg=1,
            seed=4,
            options=NoiseInversionOptions(method=method, resolution_scale=1, steps=4),
        )
        assert len(result.stages) == 1
        assert result.stages[0].latent_shape == (1, 2, 7, 9)
        assert result.stages[0].evaluations == (4 if method == "euler" else 8)
        initial = sampling.noise_scaling(
            torch.tensor(0.0001), torch.zeros_like(latent), latent
        )
        expected = initial + 2 * (0.5 - 0.0001)
        assert torch.allclose(
            sampling.noise_scaling(torch.tensor(0.5), result.noise.clone(), latent),
            expected,
            atol=1e-6,
        )
    assert len(guider_calls) == 2


def test_coarse_only_inversion_reaches_forward_target(
    guider_calls: list[dict[str, Any]],
) -> None:
    """Omit a finishing stage without stopping short of the required noise level."""
    result = noise_inversion.invert_sampling_noise(
        model=_model(_flow()),
        latent=torch.ones((1, 1, 8, 12)),
        forward_sigmas=torch.tensor([0.5, 0]),
        positive=[],
        negative=None,
        cfg=1,
        seed=1,
        options=NoiseInversionOptions(finishing_steps=0),
    )
    assert len(result.stages) == len(guider_calls) == 1
    assert float(guider_calls[0]["sigmas"][-1]) == 0.5


@pytest.mark.parametrize(
    ("sampling", "target"),
    [(_flow(), 1), (_flow(), 0), (_epsilon(), 10), (_epsilon(), 11)],
)
def test_singular_or_full_denoise_is_rejected_before_guider_execution(
    guider_calls: list[dict[str, Any]], sampling: Any, target: float
) -> None:
    """Do not invent source-preserving inversion at unsupported noise endpoints."""
    with pytest.raises(ValueError, match="denoise"):
        noise_inversion.invert_sampling_noise(
            model=_model(sampling),
            latent=torch.ones((1, 1, 8, 12)),
            forward_sigmas=torch.tensor([target, 0]),
            positive=[],
            negative=None,
            cfg=1,
            seed=1,
            options=NoiseInversionOptions(),
        )
    assert guider_calls == []


def test_every_invocation_pays_inversion_cost_without_reusing_prior_noise(
    guider_calls: list[dict[str, Any]],
) -> None:
    """Keep production work honest even for two identical test invocations."""
    results = [
        noise_inversion.invert_sampling_noise(
            model=_model(_flow()),
            latent=torch.ones((1, 1, 8, 12)),
            forward_sigmas=torch.tensor([0.5, 0]),
            positive=[],
            negative=None,
            cfg=1,
            seed=1,
            options=NoiseInversionOptions(),
        )
        for _ in range(2)
    ]
    assert len(guider_calls) == 4
    assert results[0].seconds == results[1].seconds == 5
    assert torch.equal(results[0].noise, results[1].noise)
