"""Prove inversion composes with both Comfy guider paths exactly once per sample."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from comfy import samplers
from comfy.model_sampling import CONST

from simple_syrup.domain.noise_inversion import NoiseInversionOptions
from simple_syrup.runtime.guided_sampling import sample_with_optional_negative


def _model() -> Any:
    """Build a dynamic external model with real Comfy flow noise scaling."""
    scaling = CONST()
    scaling.sigma_max = 1.0
    return SimpleNamespace(
        load_device=torch.device("cpu"),
        get_model_object=lambda name: scaling,
        model=SimpleNamespace(process_latent_in=lambda x: x),
    )


@pytest.fixture
def execution_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace expensive neural execution, not the shared guider or inversion logic."""
    calls: list[dict[str, Any]] = []

    def execute(
        model: Any,
        noise: torch.Tensor,
        latent: torch.Tensor,
        sampler: Any,
        sigmas: torch.Tensor,
        **extra: Any,
    ) -> torch.Tensor:
        """Run custom inversion samplers and record the final supplied forward noise."""
        calls.append({"noise": noise.clone(), "shape": tuple(latent.shape), **extra})
        if not isinstance(sampler, samplers.KSAMPLER):
            return noise
        state = model.get_model_object("model_sampling").noise_scaling(
            sigmas[0], noise, latent, max_denoise=False
        )

        def predict(
            x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            """Return a known constant velocity through the dynamic model boundary."""
            return x - sigma.reshape((-1,) + (1,) * (x.ndim - 1)) * 0.25

        return cast(
            torch.Tensor,
            sampler.sampler_function(
                predict, state, sigmas, extra_args={}, callback=None, disable=True
            ),
        )

    def sample_custom(
        model: Any,
        noise: torch.Tensor,
        cfg: float,
        sampler: Any,
        sigmas: torch.Tensor,
        positive: Any,
        negative: Any,
        latent: torch.Tensor,
        **extra: Any,
    ) -> torch.Tensor:
        """Represent the CFG execution boundary with both original branches."""
        return execute(
            model,
            noise,
            latent,
            sampler,
            sigmas,
            positive=positive,
            negative=negative,
            cfg=cfg,
            **extra,
        )

    class PositiveOnlyGuider:
        """Represent only the installed guider's external sampling contract."""

        def __init__(self, model: Any) -> None:
            """Retain the requested model for positive-only neural execution."""
            self.model = model
            self.conds: dict[str, Any] = {}

        def inner_set_conds(self, conds: dict[str, Any]) -> None:
            """Record actual branches selected by the shared production guider."""
            self.conds = conds

        def sample(
            self,
            noise: torch.Tensor,
            latent: torch.Tensor,
            sampler: Any,
            sigmas: torch.Tensor,
            **extra: Any,
        ) -> torch.Tensor:
            """Run the requested stage with only registered positive conditioning."""
            return execute(
                self.model, noise, latent, sampler, sigmas, **self.conds, **extra
            )

    from comfy import sample

    monkeypatch.setattr(sample, "sample_custom", sample_custom)
    monkeypatch.setattr(samplers, "CFGGuider", PositiveOnlyGuider)
    return calls


@pytest.mark.parametrize("negative", [None, []])
def test_inversion_and_forward_sampling_keep_the_same_guidance_branches(
    execution_calls: list[dict[str, Any]],
    negative: Any,
) -> None:
    """Execute two inversion stages and one forward pass without recursive inversion."""
    from comfy import sample

    positive = [[torch.ones((1, 1, 1)), {}]]
    random_noise = torch.full((1, 4, 8, 12), -3.0)
    result = sample_with_optional_negative(
        comfy_sample=sample,
        model=_model(),
        noise=random_noise,
        cfg=7.5,
        sampler=object(),
        sigmas=torch.tensor([0.5, 0]),
        positive=positive,
        negative=negative,
        latent_image=torch.zeros_like(random_noise),
        seed=42,
        noise_inversion=NoiseInversionOptions(),
    )
    assert len(execution_calls) == 3
    assert [call["shape"] for call in execution_calls] == [
        (1, 4, 4, 6),
        (1, 4, 8, 12),
        (1, 4, 8, 12),
    ]
    for call in execution_calls:
        assert call["positive"] is positive
        assert call["seed"] == 42
        if negative is None:
            assert "negative" not in call and "cfg" not in call
        else:
            assert call["negative"] is negative and call["cfg"] == 7.5
    assert torch.allclose(result, torch.full_like(result, 0.24995), atol=1e-6)
    assert torch.equal(execution_calls[-1]["noise"], result)
    assert torch.equal(random_noise, torch.full_like(random_noise, -3))


@pytest.mark.parametrize("negative", [None, []])
def test_absent_inversion_preserves_supplied_noise_without_extra_execution(
    execution_calls: list[dict[str, Any]],
    negative: Any,
) -> None:
    """Keep existing workflows on their unchanged sampling path."""
    from comfy import sample

    noise = torch.ones((1, 4, 8, 12))
    result = sample_with_optional_negative(
        comfy_sample=sample,
        model=_model(),
        noise=noise,
        cfg=7.5,
        sampler=object(),
        sigmas=torch.tensor([0.5, 0]),
        positive=[],
        negative=negative,
        latent_image=torch.zeros_like(noise),
        seed=42,
    )
    assert len(execution_calls) == 1
    assert torch.equal(result, noise)
