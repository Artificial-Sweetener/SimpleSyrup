# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file are adapted from RES4LYF and AUTOMATIC1111 scheduler
# behavior. See third_party/manifest.toml and third_party/NOTICE.md.

"""Scheduler sigma policies for SimpleSyrup sampling nodes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
from types import ModuleType
from typing import Protocol, cast

import torch

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)

EXTRA_SCHEDULERS = ("AYS SD1", "AYS SDXL", "GITS", "beta57", "automatic_a1111")
GITS_DEFAULT_COEFF = 1.20
BETA57_ALPHA = 0.5
BETA57_BETA = 0.7

AYS_NOISE_LEVELS: dict[str, tuple[float, ...]] = {
    "SD1": (
        14.6146412293,
        6.4745760956,
        3.8636745985,
        2.6946151520,
        1.8841921177,
        1.3943805092,
        0.9642583904,
        0.6523686016,
        0.3977456272,
        0.1515232662,
        0.0291671582,
    ),
    "SDXL": (
        14.6146412293,
        6.3184485287,
        3.7681790315,
        2.1811480769,
        1.3405244945,
        0.8620721141,
        0.5550693289,
        0.3798540708,
        0.2332364134,
        0.1114188177,
        0.0291671582,
    ),
}

GITS_DEFAULT_NOISE_LEVELS: tuple[tuple[float, ...], ...] = (
    (14.61464119, 0.803307, 0.02916753),
    (14.61464119, 1.56271636, 0.52423614, 0.02916753),
    (14.61464119, 2.36326075, 0.92192322, 0.36617002, 0.02916753),
    (14.61464119, 2.84484982, 1.24153244, 0.59516323, 0.25053367, 0.02916753),
    (
        14.61464119,
        5.85520077,
        2.05039096,
        0.95350921,
        0.45573691,
        0.17026083,
        0.02916753,
    ),
    (
        14.61464119,
        5.85520077,
        2.45070267,
        1.24153244,
        0.64427125,
        0.29807833,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        5.85520077,
        2.45070267,
        1.36964464,
        0.803307,
        0.45573691,
        0.25053367,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        5.85520077,
        2.84484982,
        1.61558151,
        0.95350921,
        0.59516323,
        0.36617002,
        0.19894916,
        0.09824532,
        0.02916753,
    ),
    (
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
        0.02916753,
    ),
    (
        14.61464119,
        5.85520077,
        2.95596409,
        1.84880662,
        1.24153244,
        0.83188516,
        0.59516323,
        0.41087446,
        0.27464288,
        0.17026083,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        5.85520077,
        3.07277966,
        1.98035145,
        1.36964464,
        0.95350921,
        0.69515091,
        0.50118381,
        0.36617002,
        0.25053367,
        0.17026083,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        6.77309084,
        3.46139455,
        2.36326075,
        1.56271636,
        1.08895338,
        0.803307,
        0.59516323,
        0.45573691,
        0.34370604,
        0.25053367,
        0.17026083,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        6.77309084,
        3.46139455,
        2.45070267,
        1.61558151,
        1.162866,
        0.86115354,
        0.64427125,
        0.50118381,
        0.38853383,
        0.29807833,
        0.22545385,
        0.17026083,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        7.49001646,
        4.65472794,
        3.07277966,
        2.12350607,
        1.51179266,
        1.08895338,
        0.83188516,
        0.64427125,
        0.50118381,
        0.38853383,
        0.29807833,
        0.22545385,
        0.17026083,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        7.49001646,
        4.65472794,
        3.07277966,
        2.12350607,
        1.51179266,
        1.08895338,
        0.83188516,
        0.64427125,
        0.50118381,
        0.41087446,
        0.32104823,
        0.25053367,
        0.19894916,
        0.13792117,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        7.49001646,
        4.65472794,
        3.07277966,
        2.12350607,
        1.51179266,
        1.08895338,
        0.83188516,
        0.64427125,
        0.50118381,
        0.41087446,
        0.34370604,
        0.27464288,
        0.22545385,
        0.17026083,
        0.13792117,
        0.09824532,
        0.02916753,
    ),
    (
        14.61464119,
        7.49001646,
        4.65472794,
        3.07277966,
        2.19988537,
        1.61558151,
        1.20157266,
        0.92192322,
        0.72133851,
        0.57119018,
        0.45573691,
        0.36617002,
        0.29807833,
        0.25053367,
        0.19894916,
        0.17026083,
        0.13792117,
        0.09824532,
        0.02916753,
    ),
    (
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
        0.4783645,
        0.38853383,
        0.32104823,
        0.27464288,
        0.22545385,
        0.19894916,
        0.17026083,
        0.13792117,
        0.09824532,
        0.02916753,
    ),
    (
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
        0.02916753,
    ),
)


class SamplingModel(Protocol):
    """Expose the ComfyUI model sampling object needed for core schedulers."""

    def get_model_object(self, name: str) -> object:
        """Return a named ComfyUI model object."""


def available_schedulers() -> tuple[str, ...]:
    """Return core ComfyUI schedulers plus locally resolved extra schedulers."""

    comfy_samplers = _comfy_samplers()
    core_schedulers = tuple(str(name) for name in comfy_samplers.KSampler.SCHEDULERS)
    return _unique_scheduler_names(core_schedulers + EXTRA_SCHEDULERS)


def calculate_sigmas(
    model: SamplingModel,
    scheduler_name: str,
    sampler_name: str,
    steps: int,
    denoise: float,
) -> torch.Tensor:
    """Calculate sigmas for a core or SimpleSyrup-owned scheduler."""

    supported_schedulers = available_schedulers()
    if scheduler_name not in supported_schedulers:
        supported = ", ".join(supported_schedulers)
        LOGGER.error(
            "Unsupported scheduler requested",
            extra={
                "operation": "calculate_sigmas",
                "scheduler_name": scheduler_name,
                "sampler_name": sampler_name,
                "supported_schedulers": supported,
            },
        )
        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. "
            f"Supported schedulers are: {supported}"
        )

    if scheduler_name in EXTRA_SCHEDULERS:
        return _calculate_extra_sigmas(
            model=model,
            scheduler_name=scheduler_name,
            sampler_name=sampler_name,
            steps=steps,
            denoise=denoise,
        )

    return _calculate_core_sigmas(
        model=model,
        scheduler_name=scheduler_name,
        sampler_name=sampler_name,
        steps=steps,
        denoise=denoise,
    )


def _calculate_core_sigmas(
    model: SamplingModel,
    scheduler_name: str,
    sampler_name: str,
    steps: int,
    denoise: float,
) -> torch.Tensor:
    """Calculate sigmas with ComfyUI's core scheduler policy."""

    if denoise <= 0.0:
        return torch.FloatTensor([])

    schedule_steps = _calculate_schedule_steps(steps, denoise)
    sigmas = _calculate_core_sigmas_for_steps(
        model=model,
        scheduler_name=scheduler_name,
        sampler_name=sampler_name,
        steps=schedule_steps,
    )
    return _apply_ksampler_denoise(sigmas, steps, denoise)


def _calculate_core_sigmas_for_steps(
    model: SamplingModel,
    scheduler_name: str,
    sampler_name: str,
    steps: int,
) -> torch.Tensor:
    """Apply ComfyUI scheduler calculation and sampler-specific sigma cleanup."""

    discard_penultimate_sigma = _discards_penultimate_sigma(sampler_name)
    calculation_steps = steps + 1 if discard_penultimate_sigma else steps
    comfy_samplers = _comfy_samplers()
    sigmas = cast(
        torch.Tensor,
        comfy_samplers.calculate_sigmas(
            model.get_model_object("model_sampling"),
            scheduler_name,
            calculation_steps,
        ),
    )
    if discard_penultimate_sigma:
        return torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def _calculate_extra_sigmas(
    model: SamplingModel,
    scheduler_name: str,
    sampler_name: str,
    steps: int,
    denoise: float,
) -> torch.Tensor:
    """Calculate sigmas for locally resolved extra scheduler policies."""

    if denoise <= 0.0:
        return torch.FloatTensor([])

    schedule_steps = _calculate_schedule_steps(steps, denoise)
    discard_penultimate_sigma = _discards_penultimate_sigma(sampler_name)
    calculation_steps = (
        schedule_steps + 1 if discard_penultimate_sigma else schedule_steps
    )
    sigmas = _calculate_extra_schedule(model, scheduler_name, calculation_steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return _apply_ksampler_denoise(sigmas, steps, denoise)


def _calculate_schedule_steps(steps: int, denoise: float) -> int:
    """Return KSampler's expanded step count for full schedule generation."""

    if denoise > 0.9999:
        return steps
    return int(steps / denoise)


def _apply_ksampler_denoise(
    sigmas: torch.Tensor,
    steps: int,
    denoise: float,
) -> torch.Tensor:
    """Apply KSampler's partial-denoise tail truncation."""

    if denoise > 0.9999:
        return sigmas
    return sigmas[-(steps + 1) :]


def _calculate_extra_schedule(
    model: SamplingModel,
    scheduler_name: str,
    steps: int,
) -> torch.Tensor:
    """Calculate a full local extra scheduler output."""

    if scheduler_name == "AYS SD1":
        return _calculate_ays_schedule("SD1", steps)
    if scheduler_name == "AYS SDXL":
        return _calculate_ays_schedule("SDXL", steps)
    if scheduler_name == "GITS":
        return _calculate_gits_schedule(steps)
    if scheduler_name == "beta57":
        return _calculate_beta57_schedule(model, steps)
    if scheduler_name == "automatic_a1111":
        return _calculate_automatic_a1111_schedule(model, steps)
    raise ValueError(f"Unsupported extra scheduler '{scheduler_name}'.")


def _calculate_ays_schedule(model_type: str, steps: int) -> torch.Tensor:
    """Calculate full AYS sigmas for the requested step count."""

    sigmas = list(AYS_NOISE_LEVELS[model_type])
    if (steps + 1) != len(sigmas):
        sigmas = _loglinear_interpolate(sigmas, steps + 1)
    sigmas[-1] = 0.0
    return torch.FloatTensor(sigmas)


def _calculate_gits_schedule(steps: int) -> torch.Tensor:
    """Calculate full GITS sigmas using default coeff 1.20."""

    if steps <= 20:
        sigmas = list(GITS_DEFAULT_NOISE_LEVELS[steps - 2])
    else:
        sigmas = _loglinear_interpolate(GITS_DEFAULT_NOISE_LEVELS[-1], steps + 1)
    sigmas[-1] = 0.0
    return torch.FloatTensor(sigmas)


def _calculate_beta57_schedule(model: SamplingModel, steps: int) -> torch.Tensor:
    """Calculate RES4LYF's vendored beta57 preset with ComfyUI beta scheduling."""

    comfy_samplers = _comfy_samplers()
    return cast(
        torch.Tensor,
        comfy_samplers.beta_scheduler(
            model.get_model_object("model_sampling"),
            steps,
            alpha=BETA57_ALPHA,
            beta=BETA57_BETA,
        ),
    )


def _calculate_automatic_a1111_schedule(
    model: SamplingModel,
    steps: int,
) -> torch.Tensor:
    """Calculate A1111's automatic k-diffusion discrete sigma schedule."""

    model_sampling = model.get_model_object("model_sampling")
    model_sigmas = getattr(model_sampling, "sigmas", None)
    sigma_converter = getattr(model_sampling, "sigma", None)
    if not isinstance(model_sigmas, torch.Tensor) or not callable(sigma_converter):
        raise ValueError(
            "automatic_a1111 requires a ComfyUI model_sampling object with "
            "tensor sigmas and a callable sigma(timestep) converter."
        )
    if len(model_sigmas) == 0:
        raise ValueError("automatic_a1111 requires at least one model sigma.")

    timesteps = torch.linspace(
        len(model_sigmas) - 1,
        0,
        steps,
        device=model_sigmas.device,
    )
    sigmas = sigma_converter(timesteps)
    if not isinstance(sigmas, torch.Tensor):
        raise ValueError(
            "automatic_a1111 model_sampling.sigma(timestep) must return a tensor."
        )
    return torch.cat([sigmas, sigmas.new_zeros([1])]).detach().cpu()


def _loglinear_interpolate(sigmas: Sequence[float], num_steps: int) -> list[float]:
    """Interpolate decreasing sigma values in log space."""

    if num_steps <= 1:
        return [float(sigmas[0])]

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


def _discards_penultimate_sigma(sampler_name: str) -> bool:
    """Return whether ComfyUI drops the penultimate sigma for this sampler."""

    comfy_samplers = _comfy_samplers()
    return sampler_name in comfy_samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS


def _unique_scheduler_names(names: Sequence[str]) -> tuple[str, ...]:
    """Return scheduler names in first-seen order without duplicates."""

    unique_names: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        unique_names.append(name)
        seen.add(name)
    return tuple(unique_names)


def _comfy_samplers() -> ModuleType:
    """Import ComfyUI samplers lazily to keep registration imports lightweight."""

    return import_module("comfy.samplers")
