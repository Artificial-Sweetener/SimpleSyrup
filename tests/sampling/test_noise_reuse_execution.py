"""Verify native residual parameterization and independent full-resolution finishing."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from comfy import sample as comfy_sample

from simple_syrup.domain.noise_reuse import NoiseReuseOptions, SamplingProvenance
from simple_syrup.runtime.native_noise_scaling import (
    noise_coefficients,
    recover_residual,
)
from simple_syrup.runtime.noise_reuse_finishing import finish_reused_state
from simple_syrup.runtime.noise_reuse_scope import (
    NoiseReuseExecution,
    current_noise_reuse,
    noise_reuse_scope,
)


@pytest.mark.parametrize("flow", (False, True))
def test_native_coefficients_reconstruct_eps_and_flow_states(flow: bool) -> None:
    """Detect treating flow noise as an additive epsilon residual."""
    clean = torch.arange(16).reshape(1, 4, 2, 2).float() / 4
    noise = torch.linspace(-2, 2, 16).reshape_as(clean)
    scaling = SimpleNamespace(
        noise_scaling=lambda sigma, residual, reference, maximum: (
            (1 - sigma) * reference + sigma * residual
            if flow
            else reference + sigma * residual
        )
    )
    state = scaling.noise_scaling(clean.new_tensor(0.4), noise, clean, False)
    recovered = recover_residual(scaling, clean, state, 0.4)
    base, amplitude = noise_coefficients(scaling, clean, 0.4)
    assert torch.allclose(recovered, noise, atol=2e-6)
    assert torch.allclose(base + amplitude * recovered, state, atol=2e-6)
    with pytest.raises(ValueError, match="invertible"):
        recover_residual(scaling, clean, state, 0)
    nonlinear = SimpleNamespace(
        noise_scaling=lambda sigma, residual, reference, maximum: (
            reference + sigma * residual.square()
        )
    )
    with pytest.raises(ValueError, match="affine"):
        noise_coefficients(nonlinear, clean, 0.4)


@pytest.mark.parametrize("method,evaluations", (("euler", 2), ("heun", 4)))
def test_finisher_consumes_supplied_state_and_native_cfg_once(
    monkeypatch: pytest.MonkeyPatch, method: Any, evaluations: int
) -> None:
    """Exercise the real integration solver without coarse inversion or double reuse."""
    initial = torch.ones(1, 4, 4, 6)
    latent = torch.full_like(initial, 7)
    model = SimpleNamespace(load_device="cpu")
    stage_model = object()
    positive, negative = object(), object()
    states: list[torch.Tensor] = []
    calls: list[dict[str, Any]] = []
    factory_shapes: list[tuple[int, ...]] = []

    def factory(value: torch.Tensor) -> Any:
        """Return one full-resolution model without resizing the clean reference."""
        factory_shapes.append(tuple(value.shape))
        return stage_model

    def sample(*args: Any, **kwargs: Any) -> torch.Tensor:
        """Stand in only for the external Comfy guider, retaining real solver math."""
        assert current_noise_reuse() is None
        assert args[0] is stage_model and args[2] == 6.0
        assert args[5] is positive and args[6] is negative and args[7] is latent
        assert torch.count_nonzero(args[1]) == 0
        calls.append(kwargs)

        def predict(x: torch.Tensor, sigma: torch.Tensor, **extra: Any) -> torch.Tensor:
            """Expose recognizable native denoised predictions for integration."""
            states.append(x.clone())
            return torch.zeros_like(x)

        result: torch.Tensor = args[3].sampler_function(
            predict, torch.full_like(initial, 99), args[4], {}, None, True
        )
        return result

    monkeypatch.setattr(comfy_sample, "sample_custom", sample)
    execution = NoiseReuseExecution(
        NoiseReuseOptions(), SamplingProvenance("SDXL", latent)
    )
    with noise_reuse_scope(execution):
        output = finish_reused_state(
            model=model,
            latent=latent,
            initial=initial,
            begin=0.3,
            target=0.4,
            options=NoiseReuseOptions(finishing_steps=2, finishing_method=method),
            positive=positive,
            negative=negative,
            cfg=6.0,
            seed=17,
            noise_mask=None,
            model_factory=factory,
        )
        assert current_noise_reuse() is execution
    assert current_noise_reuse() is None
    assert len(states) == evaluations and len(calls) == 1
    assert torch.equal(states[0], initial)
    assert torch.allclose(output, initial * (0.4 / 0.3), atol=2e-6)
    assert factory_shapes == [tuple(latent.shape)]
    assert calls[0]["seed"] == 17


def test_scope_restores_receiver_after_nested_failure() -> None:
    """Prevent failed finishing from leaving another workflow's reuse context active."""
    execution = NoiseReuseExecution(
        NoiseReuseOptions(), SamplingProvenance("SDXL", torch.zeros(1, 4, 2, 2))
    )
    with noise_reuse_scope(execution):
        with pytest.raises(RuntimeError, match="boundary"):
            with noise_reuse_scope(None):
                raise RuntimeError("boundary")
        assert current_noise_reuse() is execution
    assert current_noise_reuse() is None
