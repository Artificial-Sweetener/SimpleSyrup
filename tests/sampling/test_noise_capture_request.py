# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify receiving prediction views and model-aware schedule geometry."""

from typing import Any, TypedDict

import pytest
import torch

from simple_syrup.domain.noise_reuse import NoiseReuseOptions
from simple_syrup.domain.sampler_options import (
    ContextualDiffusionOptions,
    SamplerOptions,
)
from simple_syrup.runtime.comfy_latent_normalization import COMFY_LATENT_NORMALIZER
from simple_syrup.runtime.sampling_schedulers import SchedulerView
from simple_syrup.services import noise_capture_request_service as requests


class CaptureArguments(TypedDict):
    """Describe graph controls at the external MODEL and LATENT boundary."""

    model: object
    steps: int
    sampler_name: str
    scheduler: str
    denoise: float
    source_latent: dict[str, Any]
    geometry_json: str


def test_cd_coordinate_ignores_unresolved_scale_and_coalesces_finish_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the CD context schedule to select source capture coordinates."""
    views: list[SchedulerView] = []

    def schedule(*args: Any, view: SchedulerView) -> torch.Tensor:
        """Represent the external native schedule boundary with a known coordinate."""
        views.append(view)
        return torch.tensor([0.4, 0.2, 0])

    monkeypatch.setattr(requests, "calculate_sigmas", schedule)
    monkeypatch.setattr(
        requests,
        "validate_partial_sampling_target",
        lambda model, sigmas, **kwargs: float(sigmas[0]),
    )
    arguments: CaptureArguments = {
        "model": object(),
        "steps": 16,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 0.2,
        "source_latent": {"samples": torch.zeros(1, 16, 1, 96, 168)},
        "geometry_json": '[{"scale": ["dynamic_factor", 0]}]',
    }
    prefix = ContextualDiffusionOptions(context_size=128)
    service = requests.NoiseCaptureRequestService()
    original = service.coordinates(
        **arguments,
        options=SamplerOptions(
            contextual_diffusion=prefix, noise_reuse=NoiseReuseOptions()
        ),
    )
    finished = service.coordinates(
        **arguments,
        previous=original,
        options=SamplerOptions(
            contextual_diffusion=prefix,
            noise_reuse=NoiseReuseOptions(finishing_steps=1),
        ),
    )
    assert finished == (
        float(torch.tensor(0.4)),
        float(torch.tensor(float(torch.tensor(0.4)) * 0.75)),
    )
    assert views == [SchedulerView(128, 128)] * 2


def test_flux2_planning_normalizes_empty_layout_before_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match native packed Flux2 dimensions before its resolution-dependent schedule."""
    views: list[SchedulerView] = []
    normalized: list[object] = []

    def normalize(**kwargs: Any) -> torch.Tensor:
        """Stand in for Comfy's empty latent normalization boundary."""
        normalized.append(kwargs["spatial_downscale_ratio"])
        return torch.zeros(1, 128, 8, 12)

    def schedule(*args: Any, view: SchedulerView) -> torch.Tensor:
        """Record the prediction canvas passed to the native scheduler."""
        views.append(view)
        return torch.tensor([0.5, 0])

    monkeypatch.setattr(COMFY_LATENT_NORMALIZER, "normalize", normalize)
    monkeypatch.setattr(requests, "calculate_sigmas", schedule)
    monkeypatch.setattr(
        requests,
        "validate_partial_sampling_target",
        lambda model, sigmas, **kwargs: float(sigmas[0]),
    )
    result = requests.NoiseCaptureRequestService().coordinates(
        model=object(),
        steps=4,
        sampler_name="euler",
        scheduler="Flux2",
        denoise=0.25,
        options=SamplerOptions(noise_reuse=NoiseReuseOptions()),
        source_latent={
            "samples": torch.zeros(1, 4, 16, 24),
            "downscale_ratio_spacial": 8,
        },
        geometry_json='[{"scale": 1.5}]',
    )
    assert result == (0.5,) and normalized == [8]
    assert views == [SchedulerView(18, 12)]


def test_stock_fixed_size_matches_comfy_pixel_and_aspect_rules() -> None:
    """Match stock resizing's eight-pixel convention and aspect handling."""
    samples = torch.zeros(1, 128, 8, 12)
    service = requests.NoiseCaptureRequestService()
    assert service._view(samples, '[{"pixels": [0, 288]}]') == SchedulerView(36, 24)
    assert service._view(samples, '[{"pixels": [0, 0]}]') == SchedulerView(12, 8)
    with pytest.raises(ValueError, match="scale"):
        service._view(samples, '[{"scale": ["dynamic", 0]}]')
