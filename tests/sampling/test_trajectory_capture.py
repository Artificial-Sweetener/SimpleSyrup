"""Exercise capture through native Euler solver states and verified endpoint
identity.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from comfy.k_diffusion.sampling import sample_euler

from simple_syrup.domain.noise_reuse import PROVENANCE_KEY, SamplingProvenance
from simple_syrup.runtime.trajectory_capture import (
    TrajectoryCapture,
    insert_capture_coordinates,
)


class NativeSampler:
    """Exercise the same sampler-instance method binding as native Comfy."""

    def __init__(self) -> None:
        """Keep native solver ownership local to the sampler instance."""
        self.sampler_function: Any = sample_euler

    def sample(self, *args: Any) -> torch.Tensor:
        """Invoke the currently owned solver using a deterministic denoised
        prediction.
        """

        def model(
            state: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            """Return a clean zero reference through the native model boundary."""
            return torch.zeros_like(state)

        result: torch.Tensor = self.sampler_function(model, *args, disable=True)
        return result


class Executor:
    """Represent Comfy's external sampler-wrapper invocation contract."""

    def __init__(self) -> None:
        """Retain one externally owned native sampler."""
        self.class_obj = NativeSampler()
        self.original: Any = self.class_obj.sample

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Bind replacement sampler methods like the native WrapperExecutor does."""
        result: torch.Tensor = self.original(*args, **kwargs)
        return result


def test_native_euler_captures_actual_state_at_inserted_coordinate() -> None:
    """Detect accidental capture of the denoised prediction instead of noisy x."""
    format_type = type("Wan21", (), {})
    model = SimpleNamespace(
        model=SimpleNamespace(
            latent_format=format_type(), process_latent_in=lambda x: x
        )
    )
    model.get_model_object = lambda name: SimpleNamespace(
        noise_scaling=lambda sigma, noise, clean, maximum: clean + sigma * noise
    )
    capture = TrajectoryCapture(model, "euler", (0.6, 0.3))
    initial = torch.ones(1, 16, 1, 3, 4)
    result = capture.observe(Executor(), initial, torch.tensor([1.0, 0.5, 0.0]))
    assert torch.allclose(capture.states[float(torch.tensor(0.6))], initial * 0.6)
    assert torch.allclose(capture.states[float(torch.tensor(0.3))], initial * 0.3)
    attached = capture.attach({"samples": result})
    provenance = attached[PROVENANCE_KEY]
    assert isinstance(provenance, SamplingProvenance)
    assert provenance.latent_format == "Wan21"
    assert torch.equal(provenance.clean, result)
    assert torch.allclose(provenance.residual_at(float(torch.tensor(0.6))), initial)
    assert initial.sum() > 0 and result.sum() == 0
    with pytest.raises(ValueError, match="endpoint"):
        capture.attach({"samples": result + 1})


def test_capture_uses_native_clean_endpoint_before_external_format_roundoff() -> None:
    """Preserve the original trajectory's reference across lossy format conversions."""
    initial = torch.ones(1, 16, 1, 3, 4)
    model = SimpleNamespace(
        model=SimpleNamespace(
            latent_format=type("Wan21", (), {})(),
            process_latent_in=lambda value: value + 1e-7,
        ),
        get_model_object=lambda name: SimpleNamespace(
            noise_scaling=lambda sigma, noise, clean, maximum: clean + sigma * noise
        ),
    )
    capture = TrajectoryCapture(model, "euler", (0.5,))
    result = capture.observe(Executor(), initial, torch.tensor([1.0, 0.5, 0.0]))
    provenance = capture.attach({"samples": result})[PROVENANCE_KEY]
    assert isinstance(provenance, SamplingProvenance)
    assert torch.equal(provenance.clean, result)
    assert torch.equal(provenance.residual_at(0.5), initial)


def test_coordinate_insertion_keeps_boundaries_and_rejects_invalid_ranges() -> None:
    """Avoid changing source schedule points or silently accepting unavailable
    noise levels.
    """
    original = torch.tensor([1.0, 0.7, 0.4, 0.0])
    schedule = insert_capture_coordinates(original, (0.6, 0.2))
    assert torch.equal(schedule, torch.tensor([1.0, 0.7, 0.6, 0.4, 0.2, 0.0]))
    assert torch.equal(original, torch.tensor([1.0, 0.7, 0.4, 0.0]))
    assert torch.equal(insert_capture_coordinates(schedule, (0.6,)), schedule)
    with pytest.raises(ValueError, match="outside"):
        insert_capture_coordinates(original, (1.5,))
    with pytest.raises(ValueError, match="leftover noise"):
        insert_capture_coordinates(torch.tensor([1.0, 0.7, 0.4]), (0.6,))


def test_incomplete_capture_and_unsupported_solver_fail_closed() -> None:
    """Keep unavailable data from masquerading as usable source noise."""
    capture = TrajectoryCapture(object(), "dpmpp_2m", (0.5,))
    with pytest.raises(ValueError, match="Euler"):
        capture.prepare_model()
