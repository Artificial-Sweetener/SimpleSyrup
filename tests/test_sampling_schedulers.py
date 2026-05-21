# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for KSampler Extras scheduler runtime helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import comfy.samplers
import pytest
import torch

from simple_syrup.runtime import sampling_schedulers
from simple_syrup.runtime.sampling_schedulers import (
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
    assert schedulers[-5:] == (
        "AYS SD1",
        "AYS SDXL",
        "GITS",
        "beta57",
        "automatic_a1111",
    )


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


def test_ays_sd1_full_schedule_matches_reference_values() -> None:
    """AYS SD1 full schedules match fixed Comfy Extras reference values."""

    model = FakeModel()

    assert_sigmas_close(
        calculate_sigmas(model, "AYS SD1", "euler", 10, 1.0),
        [
            14.61464119,
            6.474576,
            3.86367464,
            2.69461513,
            1.88419211,
            1.39438045,
            0.96425837,
            0.65236861,
            0.39774564,
            0.15152326,
            0.0,
        ],
    )
    assert_sigmas_close(
        calculate_sigmas(model, "AYS SD1", "euler", 20, 1.0),
        [
            14.61464119,
            9.72746658,
            6.474576,
            5.00156546,
            3.86367464,
            3.22662616,
            2.69461513,
            2.25325823,
            1.88419211,
            1.62088883,
            1.39438045,
            1.15954435,
            0.96425837,
            0.79312789,
            0.65236861,
            0.50938863,
            0.39774564,
            0.24549484,
            0.15152326,
            0.06647934,
            0.0,
        ],
    )


def test_ays_sdxl_full_schedule_matches_reference_values() -> None:
    """AYS SDXL full schedules match fixed Comfy Extras reference values."""

    model = FakeModel()

    assert_sigmas_close(
        calculate_sigmas(model, "AYS SDXL", "euler", 10, 1.0),
        [
            14.61464119,
            6.31844854,
            3.76817894,
            2.18114805,
            1.34052444,
            0.86207211,
            0.55506933,
            0.37985408,
            0.23323642,
            0.11141882,
            0.0,
        ],
    )
    assert_sigmas_close(
        calculate_sigmas(model, "AYS SDXL", "euler", 20, 1.0),
        [
            14.61464119,
            9.60946751,
            6.31844854,
            4.87945127,
            3.76817894,
            2.86687231,
            2.18114805,
            1.70993638,
            1.34052444,
            1.07500172,
            0.86207211,
            0.69174403,
            0.55506933,
            0.45917898,
            0.37985408,
            0.29765046,
            0.23323642,
            0.16120461,
            0.11141882,
            0.05700676,
            0.0,
        ],
    )


def test_gits_full_schedule_matches_reference_values() -> None:
    """GITS full schedules match fixed reference values at the default coefficient."""

    model = FakeModel()

    assert_sigmas_close(
        calculate_sigmas(model, "GITS", "euler", 10, 1.0),
        [
            14.61464119,
            5.85520077,
            2.84484982,
            1.67050016,
            1.08895338,
            0.74807048,
            0.50118381,
            0.32104823,
            0.19894916,
            0.09824532,
            0.0,
        ],
    )
    assert_sigmas_close(
        calculate_sigmas(model, "GITS", "euler", 20, 1.0),
        [
            14.61464119,
            7.49001646,
            4.65472794,
            3.07277966,
            2.19988537,
            1.61558151,
            1.24153244,
            0.95350921,
            0.74807048,
            0.59516323,
            0.50118381,
            0.41087446,
            0.34370604,
            0.29807833,
            0.25053367,
            0.22545385,
            0.19894916,
            0.17026083,
            0.13792117,
            0.09824532,
            0.0,
        ],
    )
    assert_sigmas_close(
        calculate_sigmas(model, "GITS", "euler", 30, 1.0),
        [
            14.61464119,
            9.35946941,
            6.39175272,
            4.65472794,
            3.52900577,
            2.74887061,
            2.19988537,
            1.7906853,
            1.47980762,
            1.24153244,
            1.04120433,
            0.87942225,
            0.74807048,
            0.64230049,
            0.56202602,
            0.50118381,
            0.43900734,
            0.38714039,
            0.34370604,
            0.31257147,
            0.28130382,
            0.25053367,
            0.23352164,
            0.21624818,
            0.19894916,
            0.17933175,
            0.1587158,
            0.13792117,
            0.11000647,
            0.0655402,
            0.0,
        ],
    )


@pytest.mark.parametrize("scheduler_name", ["AYS SD1", "AYS SDXL", "GITS"])
@pytest.mark.parametrize("sampler_name", ["euler", "dpm_2"])
@pytest.mark.parametrize("steps", [5, 10, 20, 30])
@pytest.mark.parametrize("denoise", [0.25, 0.5, 0.8, 1.0])
def test_extra_schedulers_follow_ksampler_denoise_semantics(
    scheduler_name: str,
    sampler_name: str,
    steps: int,
    denoise: float,
) -> None:
    """Extra schedulers use KSampler partial-denoise and sigma cleanup rules."""

    actual = calculate_sigmas(
        FakeModel(),
        scheduler_name,
        sampler_name,
        steps,
        denoise,
    )
    expected = reference_extra_sigmas(scheduler_name, sampler_name, steps, denoise)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    if denoise < 1.0:
        assert actual.shape == (steps + 1,)


def test_reported_ays_sd1_partial_denoise_regression() -> None:
    """AYS SD1 with partial denoise returns a KSampler-length sigma schedule."""

    actual = calculate_sigmas(FakeModel(), "AYS SD1", "euler", 5, 0.25)
    expected = reference_extra_sigmas("AYS SD1", "euler", 5, 0.25)

    assert actual.shape == (6,)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_gits_scope_uses_default_coefficient() -> None:
    """The simple node contract exposes GITS at its default coefficient only."""

    assert sampling_schedulers.GITS_DEFAULT_COEFF == 1.20


def test_beta57_scope_uses_res4lyf_preset_parameters() -> None:
    """The beta57 scheduler contract exposes RES4LYF's fixed beta preset."""

    assert sampling_schedulers.BETA57_ALPHA == 0.5
    assert sampling_schedulers.BETA57_BETA == 0.7
