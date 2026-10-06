# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove observational capture preserves native sampling and real coordinates."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from comfy.k_diffusion.sampling import sample_euler

from simple_syrup.domain.noise_reuse import PROVENANCE_KEY, SamplingProvenance
from simple_syrup.runtime.sampling_samplers import available_samplers, resolve_sampler
from simple_syrup.runtime.trajectory_capture import TrajectoryCapture


class PredictionModel:
    """Emulate the native prediction boundary without loading model weights."""

    def __init__(self, capture: TrajectoryCapture | None = None) -> None:
        """Retain the optional observer around the same deterministic prediction."""
        self.capture = capture

    def __call__(
        self, state: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Run the source observation before its unchanged denoising prediction."""
        if self.capture is not None:
            return self.capture.observe_prediction(self.predict, state, sigma, **kwargs)
        return self.predict(state, sigma, **kwargs)

    def predict(
        self, state: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Return a zero endpoint with a known linear Euler trajectory."""
        return torch.zeros_like(state)


class Executor:
    """Represent Comfy's sampler wrapper while retaining the original algorithm."""

    def __init__(self, capture: TrajectoryCapture | None = None) -> None:
        """Observe only model calls, leaving callbacks and the solver untouched."""
        self.model = PredictionModel(capture)

    def __call__(
        self, state: torch.Tensor, schedule: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Execute the real Euler implementation on the supplied schedule."""
        result: torch.Tensor = sample_euler(
            self.model, state, schedule, disable=True, **kwargs
        )
        return result


def model_surface(roundoff: float = 0.0) -> Any:
    """Provide native latent format and additive noise parameterization."""
    return SimpleNamespace(
        model=SimpleNamespace(
            latent_format=type("Wan21", (), {})(),
            process_latent_in=lambda value: value + roundoff,
        ),
        get_model_object=lambda name: SimpleNamespace(
            noise_scaling=lambda sigma, noise, clean, maximum: clean + sigma * noise
        ),
    )


def test_native_euler_captures_nearest_real_coordinate_without_inserting_steps() -> (
    None
):
    """Distinguish requested levels from actually observed source states."""
    capture = TrajectoryCapture(model_surface(), "euler", (0.6, 0.3))
    initial = torch.ones(1, 16, 1, 3, 4)
    schedule = torch.tensor([1.0, 0.5, 0.0])
    result = capture.observe(Executor(capture), initial, schedule)
    assert capture.schedule is not None
    assert torch.equal(capture.schedule, schedule)
    for observation in capture.states.values():
        assert observation.coordinate == 0.5
        assert torch.equal(observation.state, initial * 0.5)
    attached = capture.attach({"samples": result})
    provenance = attached[PROVENANCE_KEY]
    assert isinstance(provenance, SamplingProvenance)
    assert provenance.latent_format == "Wan21"
    assert torch.equal(provenance.clean, result)
    for request in capture.coordinates:
        assert provenance.capture_at(request).actual_coordinate == 0.5
        assert torch.equal(provenance.residual_at(request), initial)
    assert initial.sum() > 0 and result.sum() == 0
    with pytest.raises(ValueError, match="different original sampler output"):
        capture.attach({"samples": result + 1})


def test_native_boundary_capture_preserves_source_result_and_callback() -> None:
    """Retain the source's exact schedule, endpoint and original callback sequence."""
    initial = torch.ones(1, 4, 3, 4)
    schedule = torch.tensor([1.0, 0.5, 0.0])
    callbacks: list[int] = []
    reference = Executor()(initial.clone(), schedule)
    capture = TrajectoryCapture(model_surface(), "euler", (0.5,))

    def callback(record: dict[str, Any]) -> None:
        """Retain the original solver's callback order and count."""
        callbacks.append(int(record["i"]))

    output = capture.observe(
        Executor(capture), initial.clone(), schedule, callback=callback
    )
    assert torch.equal(output, reference)
    assert callbacks == [0, 1]
    assert capture.states[0.5].coordinate == 0.5
    assert torch.equal(capture.states[0.5].state, initial * 0.5)
    assert torch.equal(schedule, torch.tensor([1.0, 0.5, 0.0]))


def test_format_only_capture_returns_the_original_model() -> None:
    """Keep Auto latent-format provenance free of numerical sampling changes."""
    model = object()
    assert TrajectoryCapture(model, "er_sde", ()).prepare_model() is model


def test_capture_uses_native_clean_endpoint_before_external_format_roundoff() -> None:
    """Preserve the original trajectory's reference across lossy format conversions."""
    initial = torch.ones(1, 16, 1, 3, 4)
    capture = TrajectoryCapture(model_surface(1e-7), "euler", (0.5,))
    result = capture.observe(Executor(capture), initial, torch.tensor([1.0, 0.5, 0.0]))
    provenance = capture.attach({"samples": result})[PROVENANCE_KEY]
    assert isinstance(provenance, SamplingProvenance)
    assert torch.equal(provenance.clean, result)
    assert torch.equal(provenance.residual_at(0.5), initial)


def test_ties_prefer_higher_noise_and_snapshots_are_immutable() -> None:
    """Make nearest selection independent of evaluation order and tensor mutation."""
    capture = TrajectoryCapture(model_surface(), "external", (0.5, 0.25))
    state = torch.ones(1, 4, 3, 4)

    def source(*args: Any) -> torch.Tensor:
        """Observe a tie followed by an exact match in an external algorithm."""
        capture.observe_prediction(
            PredictionModel().predict, state, torch.tensor([0.25])
        )
        state.mul_(3)
        capture.observe_prediction(
            PredictionModel().predict, state, torch.tensor([0.75])
        )
        state.zero_()
        return state

    capture.observe(source, state, torch.tensor([1.0, 0.0]))
    assert capture.states[0.5].coordinate == 0.75
    assert capture.states[0.5].state.mean() == 3
    assert capture.states[0.25].coordinate == 0.25
    assert capture.states[0.25].state.mean() == 1


def test_incomplete_capture_fails_closed_with_chosen_algorithm() -> None:
    """Reject a source that bypasses the native model-prediction contract."""
    capture = TrajectoryCapture(model_surface(), "external_unobserved", (0.5,))
    with pytest.raises(
        ValueError, match="external_unobserved.*is not supported.*noisy latent"
    ):
        capture.attach({"samples": torch.ones(1, 4, 3, 4)})


@pytest.mark.parametrize("coordinate", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_capture_requests_fail_before_sampling(coordinate: float) -> None:
    """Reject undefined native noise levels before installing observers."""
    with pytest.raises(ValueError, match="finite and above zero"):
        TrajectoryCapture(model_surface(), "euler", (coordinate,))


class SolverModel(PredictionModel):
    """Supply native sampler metadata and CFG hooks around a controlled model."""

    def __init__(self, capture: TrajectoryCapture) -> None:
        """Model the external guider boundary shared by all advertised algorithms."""
        super().__init__(capture)
        sampling = SimpleNamespace(
            sigma_max=torch.tensor(1.0),
            sigma_min=torch.tensor(0.01),
            noise_scaling=lambda sigma, noise, clean: clean + sigma * noise,
            calculate_denoised=lambda sigma, eps, x: x - sigma * eps,
            percent_to_sigma=lambda fraction: 1.0 - fraction,
        )
        self.inner_model = SimpleNamespace(
            inner_model=SimpleNamespace(
                device=torch.device("cpu"),
                model_sampling=sampling,
                diffusion_model=SimpleNamespace(),
            ),
            model_patcher=SimpleNamespace(get_model_object=lambda name: sampling),
        )

    def predict(
        self, state: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Run native post-CFG hooks to expose conditional and unconditional results."""
        denoised = torch.full_like(state, 0.2)
        for hook in kwargs.get("model_options", {}).get(
            "sampler_post_cfg_function", ()
        ):
            denoised = hook(
                {
                    "denoised": denoised,
                    "uncond_denoised": denoised,
                    "cond_denoised": denoised,
                }
            )
        return denoised


@pytest.mark.parametrize("sampler_name", available_samplers())
@pytest.mark.filterwarnings(
    "ignore:To copy construct from a tensor:UserWarning:comfy.k_diffusion.sampling"
)
@pytest.mark.filterwarnings(
    "ignore:__array_wrap__ must accept context:DeprecationWarning:"
    "comfy.k_diffusion.deis"
)
def test_every_advertised_algorithm_captures_real_source_noise(
    sampler_name: str,
) -> None:
    """Discover newly added algorithms automatically and require executable capture."""
    capture = TrajectoryCapture(model_surface(), sampler_name, (0.4,))
    model = SolverModel(capture)
    sampler = resolve_sampler(sampler_name)
    sample = vars(sampler)["sampler_function"]
    initial = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(19))
    schedule = torch.tensor([1.0, 0.5, 0.0])
    if sampler_name in {"uni_pc", "uni_pc_bh2"}:
        schedule = torch.tensor([1.0, 0.5, 0.1, 0.0])
    supplied_schedule = schedule.clone()

    def execute(state: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """Invoke each genuine solver with its own declared options and fixed seed."""
        result: torch.Tensor = sample(
            model,
            state,
            sigmas,
            extra_args={"seed": 42, "model_options": {"transformer_options": {}}},
            callback=None,
            disable=True,
            **vars(sampler)["extra_options"],
        )
        return result

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        reference = execute(initial.clone(), schedule.clone())
        torch.manual_seed(43)
        output = capture.observe(execute, initial.clone(), schedule)
    observation = capture.states[float(torch.tensor(0.4))]
    assert observation.coordinate > 0
    assert observation.state.shape == initial.shape
    assert torch.isfinite(observation.state).all()
    assert capture.clean is not None
    assert capture.schedule is not None
    assert torch.equal(capture.clean, output.float().cpu())
    assert torch.equal(output, reference)
    assert torch.equal(capture.schedule, supplied_schedule)
