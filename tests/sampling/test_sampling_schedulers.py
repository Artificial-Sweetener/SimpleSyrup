# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for KSampler Extras scheduler runtime helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
from typing import Any

import comfy.samplers
import pytest
import torch

from simple_syrup.runtime import sampling_schedulers
from simple_syrup.runtime.sampling_schedulers import (
    SchedulerView,
    available_schedulers,
    calculate_sigmas,
)


class FakeModel:
    """Provide the model-sampling object expected by scheduler helpers."""

    def __init__(self) -> None:
        """Create a fake model with a stable model_sampling object."""

        self.model_sampling = object()

    def get_model_object(self, name: str) -> object:
        """Return the requested fake model object."""

        assert name == "model_sampling"
        return self.model_sampling


class FakeLatentFormat:
    """Expose the spatial compression used to recover image dimensions."""

    def __init__(self, spacial_downscale_ratio: int) -> None:
        """Store one deterministic latent-to-image scale."""

        self.spacial_downscale_ratio = spacial_downscale_ratio


class FakeModelWithLatentFormat(FakeModel):
    """Provide model-sampling and latent-format objects for Flux2 tests."""

    def __init__(self, spacial_downscale_ratio: int) -> None:
        """Create a fake model with the requested spatial compression."""

        super().__init__()
        self.latent_format = FakeLatentFormat(spacial_downscale_ratio)

    def get_model_object(self, name: str) -> object:
        """Return the requested fake model object."""

        if name == "latent_format":
            return self.latent_format
        return super().get_model_object(name)


class FakeDiscreteModelSampling:
    """Provide k-diffusion-style discrete sigma conversion for tests."""

    def __init__(self, sigmas: Sequence[float]) -> None:
        """Create a fake discrete model sampling object."""

        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.log_sigmas = self.sigmas.log()

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert fractional timesteps to sigmas with log-linear interpolation."""

        timestep = torch.clamp(
            timestep.float().to(self.log_sigmas.device),
            min=0,
            max=(len(self.sigmas) - 1),
        )
        low_index = timestep.floor().long()
        high_index = timestep.ceil().long()
        weight = timestep.frac()
        log_sigma = (1 - weight) * self.log_sigmas[
            low_index
        ] + weight * self.log_sigmas[high_index]
        return log_sigma.exp().to(timestep.device)


class FakeDiscreteModel:
    """Provide a discrete model_sampling object for automatic_a1111 tests."""

    def __init__(self) -> None:
        """Create a fake model with k-diffusion-style sigmas."""

        self.model_sampling = FakeDiscreteModelSampling((0.1, 0.3, 1.0))

    def get_model_object(self, name: str) -> FakeDiscreteModelSampling:
        """Return the requested fake model sampling object."""

        assert name == "model_sampling"
        return self.model_sampling


def assert_sigmas_close(actual: torch.Tensor, expected: list[float]) -> None:
    """Assert that calculated sigmas match fixed reference values."""

    assert torch.allclose(
        actual.cpu(),
        torch.tensor(expected, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def reference_extra_sigmas(
    scheduler_name: str,
    sampler_name: str,
    steps: int,
    denoise: float,
) -> torch.Tensor:
    """Calculate expected extra scheduler sigmas with KSampler semantics."""

    if denoise <= 0.0:
        return torch.FloatTensor([])

    schedule_steps = steps if denoise > 0.9999 else int(steps / denoise)
    if sampler_name in comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
        schedule_steps += 1

    sigmas = reference_full_extra_schedule(scheduler_name, schedule_steps)
    if sampler_name in comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

    if denoise <= 0.9999:
        sigmas = sigmas[-(steps + 1) :]
    return sigmas


def reference_full_extra_schedule(scheduler_name: str, steps: int) -> torch.Tensor:
    """Calculate expected full AYS/GITS formula output for tests."""

    if scheduler_name == "AYS SD1":
        return reference_ays_schedule("SD1", steps)
    if scheduler_name == "AYS SDXL":
        return reference_ays_schedule("SDXL", steps)
    if scheduler_name == "GITS":
        return reference_gits_schedule(steps)
    if scheduler_name == "automatic_a1111":
        return reference_automatic_a1111_schedule(FakeDiscreteModel(), steps)
    raise ValueError(f"Unsupported reference scheduler '{scheduler_name}'.")


def reference_ays_schedule(model_type: str, steps: int) -> torch.Tensor:
    """Calculate full AYS schedule with Comfy Extras formula semantics."""

    sigmas = list(sampling_schedulers.AYS_NOISE_LEVELS[model_type])
    if (steps + 1) != len(sigmas):
        sigmas = reference_loglinear_interpolate(sigmas, steps + 1)
    sigmas[-1] = 0.0
    return torch.FloatTensor(sigmas)


def reference_gits_schedule(steps: int) -> torch.Tensor:
    """Calculate full GITS schedule for the default coefficient."""

    if steps <= 20:
        sigmas = list(sampling_schedulers.GITS_DEFAULT_NOISE_LEVELS[steps - 2])
    else:
        sigmas = reference_loglinear_interpolate(
            sampling_schedulers.GITS_DEFAULT_NOISE_LEVELS[-1],
            steps + 1,
        )
    sigmas[-1] = 0.0
    return torch.FloatTensor(sigmas)


def reference_automatic_a1111_schedule(
    model: FakeDiscreteModel,
    steps: int,
) -> torch.Tensor:
    """Calculate k-diffusion DiscreteSchedule.get_sigmas-style output."""

    model_sampling = model.model_sampling
    timesteps = torch.linspace(
        len(model_sampling.sigmas) - 1,
        0,
        steps,
        device=model_sampling.sigmas.device,
    )
    sigmas = model_sampling.sigma(timesteps)
    return torch.cat([sigmas, sigmas.new_zeros([1])]).cpu()


def reference_loglinear_interpolate(
    sigmas: Sequence[float],
    num_steps: int,
) -> list[float]:
    """Interpolate reference sigma values in log space."""

    reversed_logs = [math.log(value) for value in reversed(sigmas)]
    source_max = len(reversed_logs) - 1
    target_max = num_steps - 1
    interpolated: list[float] = []

    for target_index in range(num_steps):
        source_position = target_index * source_max / target_max
        left_index = math.floor(source_position)
        right_index = min(left_index + 1, source_max)
        fraction = source_position - left_index
        left_value = reversed_logs[left_index]
        right_value = reversed_logs[right_index]
        interpolated.append(
            math.exp(left_value + (right_value - left_value) * fraction)
        )

    return list(reversed(interpolated))


def test_available_schedulers_includes_core_and_extras() -> None:
    """Scheduler options combine ComfyUI core names with SimpleSyrup extras."""

    schedulers = available_schedulers()

    for scheduler in comfy.samplers.KSampler.SCHEDULERS:
        assert scheduler in schedulers
    assert schedulers[-6:] == (
        "AYS SD1",
        "AYS SDXL",
        "GITS",
        "beta57",
        "automatic_a1111",
        "Flux2",
    )


def test_flux2_schedule_matches_comfy_for_model_view_resolution() -> None:
    """Flux2 delegates to ComfyUI using the effective model-view resolution."""

    model = FakeModelWithLatentFormat(spacial_downscale_ratio=16)
    sigmas = calculate_sigmas(
        model,
        "Flux2",
        "euler",
        4,
        1.0,
        view=SchedulerView(latent_width=64, latent_height=64),
    )

    flux_nodes = import_module("comfy_extras.nodes_flux")
    expected = torch.as_tensor(flux_nodes.get_schedule(4, 4096), dtype=torch.float32)
    assert torch.allclose(sigmas, expected, atol=1e-6, rtol=1e-6)


def test_flux2_schedule_requires_a_model_view() -> None:
    """Flux2 fails clearly when a caller omits its resolution context."""

    with pytest.raises(ValueError, match="requires a model view"):
        calculate_sigmas(FakeModel(), "Flux2", "euler", 4, 1.0)


def test_flux2_schedule_uses_model_latent_downscale_ratio() -> None:
    """Flux2 remains selectable for models with non-Flux latent formats."""

    model = FakeModelWithLatentFormat(spacial_downscale_ratio=8)
    sigmas = calculate_sigmas(
        model,
        "Flux2",
        "euler",
        4,
        1.0,
        view=SchedulerView(latent_width=128, latent_height=128),
    )

    flux_nodes = import_module("comfy_extras.nodes_flux")
    expected = torch.as_tensor(flux_nodes.get_schedule(4, 4096), dtype=torch.float32)
    assert torch.allclose(sigmas, expected, atol=1e-6, rtol=1e-6)


def test_available_schedulers_deduplicates_beta57_when_globally_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local beta57 option is shown once if another extension patched ComfyUI."""

    monkeypatch.setattr(
        comfy.samplers.KSampler,
        "SCHEDULERS",
        tuple(comfy.samplers.KSampler.SCHEDULERS) + ("beta57",),
    )

    schedulers = available_schedulers()

    assert schedulers.count("beta57") == 1
    assert "beta57" in schedulers


def test_available_schedulers_deduplicates_automatic_a1111_when_globally_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local A1111 scheduler is shown once if another extension patched ComfyUI."""

    monkeypatch.setattr(
        comfy.samplers.KSampler,
        "SCHEDULERS",
        tuple(comfy.samplers.KSampler.SCHEDULERS) + ("automatic_a1111",),
    )

    schedulers = available_schedulers()

    assert schedulers.count("automatic_a1111") == 1
    assert "automatic_a1111" in schedulers


def test_available_schedulers_excludes_svd_scheduler() -> None:
    """Unsupported SVD scheduling is excluded from the available scheduler list."""

    assert "AYS SVD" not in available_schedulers()


def test_unknown_scheduler_is_rejected() -> None:
    """Unsupported scheduler names fail before sampling begins."""

    with pytest.raises(ValueError, match="Unsupported scheduler 'not-real'"):
        calculate_sigmas(FakeModel(), "not-real", "euler", 20, 1.0)


def test_core_scheduler_delegates_to_comfy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core schedulers use ComfyUI's installed sigma implementation."""

    calls: list[dict[str, Any]] = []

    def fake_calculate_sigmas(
        model_sampling: object,
        scheduler_name: str,
        steps: int,
    ) -> torch.Tensor:
        """Record delegation arguments and return deterministic sigmas."""

        calls.append(
            {
                "model_sampling": model_sampling,
                "scheduler_name": scheduler_name,
                "steps": steps,
            }
        )
        return torch.arange(steps + 1, dtype=torch.float32)

    monkeypatch.setattr(
        comfy.samplers,
        "calculate_sigmas",
        fake_calculate_sigmas,
    )

    model = FakeModel()
    sigmas = calculate_sigmas(model, "normal", "euler", 4, 1.0)

    assert calls == [
        {
            "model_sampling": model.model_sampling,
            "scheduler_name": "normal",
            "steps": 4,
        }
    ]
    assert torch.equal(sigmas, torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))


def test_core_scheduler_zero_denoise_returns_empty_tensor() -> None:
    """Denoise zero skips sigma generation."""

    sigmas = calculate_sigmas(FakeModel(), "normal", "euler", 20, 0.0)

    assert sigmas.shape == (0,)


def test_core_scheduler_partial_denoise_truncates_sigmas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial denoise follows built-in KSampler truncation semantics."""

    def fake_calculate_sigmas(
        model_sampling: object,
        scheduler_name: str,
        steps: int,
    ) -> torch.Tensor:
        """Return a predictable sequence for denoise truncation."""

        return torch.arange(steps + 1, dtype=torch.float32)

    monkeypatch.setattr(
        comfy.samplers,
        "calculate_sigmas",
        fake_calculate_sigmas,
    )

    sigmas = calculate_sigmas(FakeModel(), "normal", "euler", 4, 0.5)

    assert torch.equal(sigmas, torch.tensor([4, 5, 6, 7, 8], dtype=torch.float32))


def test_beta57_full_denoise_uses_res4lyf_preset_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """beta57 calls ComfyUI's beta scheduler with RES4LYF's vendored preset."""

    calls: list[dict[str, object]] = []

    def fake_beta_scheduler(
        model_sampling: object,
        steps: int,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        """Record beta scheduler arguments and return deterministic sigmas."""

        calls.append(
            {
                "model_sampling": model_sampling,
                "steps": steps,
                "alpha": alpha,
                "beta": beta,
            }
        )
        return torch.arange(steps + 1, dtype=torch.float32)

    monkeypatch.setattr(comfy.samplers, "beta_scheduler", fake_beta_scheduler)

    model = FakeModel()
    sigmas = calculate_sigmas(model, "beta57", "euler", 4, 1.0)

    assert calls == [
        {
            "model_sampling": model.model_sampling,
            "steps": 4,
            "alpha": sampling_schedulers.BETA57_ALPHA,
            "beta": sampling_schedulers.BETA57_BETA,
        }
    ]
    assert torch.equal(sigmas, torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))


def test_beta57_partial_denoise_uses_expanded_schedule_then_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """beta57 follows KSampler partial-denoise expansion and tail truncation."""

    calls: list[int] = []

    def fake_beta_scheduler(
        model_sampling: object,
        steps: int,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        """Return a predictable sequence for denoise truncation."""

        del model_sampling, alpha, beta
        calls.append(steps)
        return torch.arange(steps + 1, dtype=torch.float32)

    monkeypatch.setattr(comfy.samplers, "beta_scheduler", fake_beta_scheduler)

    sigmas = calculate_sigmas(FakeModel(), "beta57", "euler", 4, 0.5)

    assert calls == [8]
    assert torch.equal(sigmas, torch.tensor([4, 5, 6, 7, 8], dtype=torch.float32))


def test_beta57_zero_denoise_returns_empty_tensor_without_scheduler_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Denoise zero skips beta57 sigma generation."""

    def fake_beta_scheduler(
        model_sampling: object,
        steps: int,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        """Fail if zero denoise reaches ComfyUI scheduler calculation."""

        del model_sampling, steps, alpha, beta
        raise AssertionError("beta_scheduler should not be called")

    monkeypatch.setattr(comfy.samplers, "beta_scheduler", fake_beta_scheduler)

    sigmas = calculate_sigmas(FakeModel(), "beta57", "euler", 20, 0.0)

    assert sigmas.shape == (0,)


def test_beta57_discards_penultimate_sigma_for_matching_samplers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """beta57 keeps ComfyUI KSampler cleanup for DPM-style sampler schedules."""

    calls: list[int] = []

    def fake_beta_scheduler(
        model_sampling: object,
        steps: int,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        """Return sigmas long enough to verify penultimate cleanup."""

        del model_sampling, alpha, beta
        calls.append(steps)
        return torch.arange(steps + 1, dtype=torch.float32)

    monkeypatch.setattr(comfy.samplers, "beta_scheduler", fake_beta_scheduler)

    sigmas = calculate_sigmas(FakeModel(), "beta57", "dpm_2", 4, 1.0)

    assert calls == [5]
    assert torch.equal(sigmas, torch.tensor([0, 1, 2, 3, 5], dtype=torch.float32))


def test_beta57_uses_local_path_when_comfy_scheduler_list_is_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """beta57 is resolved locally even if another extension patched ComfyUI."""

    def fail_core_calculate_sigmas(
        model_sampling: object,
        scheduler_name: str,
        steps: int,
    ) -> torch.Tensor:
        """Fail if beta57 delegates to ComfyUI's global scheduler lookup."""

        del model_sampling, scheduler_name, steps
        raise AssertionError("beta57 should not use core calculate_sigmas")

    def fake_beta_scheduler(
        model_sampling: object,
        steps: int,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        """Return deterministic local beta57 sigmas."""

        del model_sampling, alpha, beta
        return torch.arange(steps + 1, dtype=torch.float32)

    monkeypatch.setattr(
        comfy.samplers.KSampler,
        "SCHEDULERS",
        tuple(comfy.samplers.KSampler.SCHEDULERS) + ("beta57",),
    )
    monkeypatch.setattr(
        comfy.samplers,
        "calculate_sigmas",
        fail_core_calculate_sigmas,
    )
    monkeypatch.setattr(comfy.samplers, "beta_scheduler", fake_beta_scheduler)

    sigmas = calculate_sigmas(FakeModel(), "beta57", "euler", 4, 1.0)

    assert torch.equal(sigmas, torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))


def test_automatic_a1111_full_denoise_matches_discrete_schedule() -> None:
    """automatic_a1111 reproduces k-diffusion DiscreteSchedule.get_sigmas."""

    model = FakeDiscreteModel()
    sigmas = calculate_sigmas(model, "automatic_a1111", "euler", 4, 1.0)
    expected = reference_automatic_a1111_schedule(model, 4)

    assert torch.allclose(sigmas, expected, atol=1e-6, rtol=1e-6)


def test_automatic_a1111_partial_denoise_expands_then_truncates() -> None:
    """automatic_a1111 follows KSampler partial-denoise expansion semantics."""

    model = FakeDiscreteModel()
    sigmas = calculate_sigmas(model, "automatic_a1111", "euler", 4, 0.5)
    expected = reference_automatic_a1111_schedule(model, 8)[-(4 + 1) :]

    assert torch.allclose(sigmas, expected, atol=1e-6, rtol=1e-6)


def test_automatic_a1111_zero_denoise_returns_empty_tensor() -> None:
    """Denoise zero skips automatic_a1111 sigma generation."""

    sigmas = calculate_sigmas(FakeDiscreteModel(), "automatic_a1111", "euler", 20, 0.0)

    assert sigmas.shape == (0,)


def test_automatic_a1111_discards_penultimate_sigma_for_matching_samplers() -> None:
    """automatic_a1111 keeps ComfyUI KSampler cleanup for DPM-style schedules."""

    model = FakeDiscreteModel()
    sigmas = calculate_sigmas(model, "automatic_a1111", "dpm_2", 4, 1.0)
    expected = reference_automatic_a1111_schedule(model, 5)
    expected = torch.cat([expected[:-2], expected[-1:]])

    assert torch.allclose(sigmas, expected, atol=1e-6, rtol=1e-6)


def test_automatic_a1111_uses_local_path_when_comfy_scheduler_list_is_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """automatic_a1111 is resolved locally even if ComfyUI is globally patched."""

    def fail_core_calculate_sigmas(
        model_sampling: object,
        scheduler_name: str,
        steps: int,
    ) -> torch.Tensor:
        """Fail if automatic_a1111 delegates to ComfyUI's scheduler lookup."""

        del model_sampling, scheduler_name, steps
        raise AssertionError("automatic_a1111 should not use core calculate_sigmas")

    monkeypatch.setattr(
        comfy.samplers.KSampler,
        "SCHEDULERS",
        tuple(comfy.samplers.KSampler.SCHEDULERS) + ("automatic_a1111",),
    )
    monkeypatch.setattr(
        comfy.samplers,
        "calculate_sigmas",
        fail_core_calculate_sigmas,
    )

    sigmas = calculate_sigmas(FakeDiscreteModel(), "automatic_a1111", "euler", 4, 1.0)
    expected = reference_automatic_a1111_schedule(FakeDiscreteModel(), 4)

    assert torch.allclose(sigmas, expected, atol=1e-6, rtol=1e-6)


def test_automatic_a1111_rejects_unsupported_model_sampling_object() -> None:
    """automatic_a1111 fails clearly for model sampling objects without sigmas."""

    with pytest.raises(ValueError, match="automatic_a1111 requires"):
        calculate_sigmas(FakeModel(), "automatic_a1111", "euler", 4, 1.0)
