# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact denoiser-sequence measurement and failure boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import comfy.model_management
import pytest
import torch

from tools.anima_regional_lora_performance.manifest import PerformanceProfile
from tools.anima_regional_lora_performance.measurement import (
    activate_measurement_model,
    execute_call_sequence,
)
from tools.anima_regional_lora_performance.runtime_profile import (
    PerformanceRuntimeProfile,
)


@dataclass(slots=True)
class _DiffusionModel:
    """Record every sampled sigma while returning deterministic tensor output."""

    calls: list[float] = field(default_factory=list)
    invalid_shape: bool = False

    def apply_model(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor,
        *,
        c_crossattn: torch.Tensor,
        transformer_options: dict[str, object],
    ) -> torch.Tensor:
        """Return a deterministic same-shape value or one deliberate bad shape."""

        assert c_crossattn.shape == (2, 1, 1)
        assert transformer_options["cond_or_uncond"] == [0, 1]
        self.calls.append(float(sigma[0].item()))
        if self.invalid_shape:
            return latent[:1]
        return latent + sigma.reshape((-1,) + (1,) * (latent.ndim - 1))


@dataclass(slots=True)
class _ModelPatcher:
    """Expose one model owner through the Comfy MODEL patcher shape."""

    model: _DiffusionModel


def test_measurement_activation_fully_loads_the_exact_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load source graphs before derivation and clones before timed execution."""

    model = object()
    captured: list[tuple[list[object], bool]] = []

    def load_models_gpu(
        models: list[object],
        *,
        force_full_load: bool,
    ) -> None:
        """Capture the installed Comfy activation request."""

        captured.append((models, force_full_load))

    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        load_models_gpu,
    )

    activate_measurement_model(model)

    assert captured == [([model], True)]


def test_sequence_measurement_records_exact_calls_memory_and_stable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure only the requested sigma prefix and preserve deterministic output."""

    _replace_cuda_measurement(monkeypatch, peak_bytes=12_345)
    diffusion = _DiffusionModel()
    profile = _profile(diffusion)
    latent = torch.zeros((2, 1, 2), dtype=torch.float32)
    context = torch.zeros((2, 1, 1), dtype=torch.float32)
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)

    first = execute_call_sequence(
        profile,
        latent=latent,
        context=context,
        sample_sigmas=sample_sigmas,
        call_count=2,
        measured=True,
    )
    second = execute_call_sequence(
        profile,
        latent=latent,
        context=context,
        sample_sigmas=sample_sigmas,
        call_count=2,
        measured=True,
    )

    assert diffusion.calls == [1.0, 0.5, 1.0, 0.5]
    assert first.denoiser_calls == 2
    assert first.peak_vram_bytes == 12_345
    assert first.runtime_ms > 0.0
    assert first.output_sha256 == second.output_sha256
    assert len(first.output_sha256) == 64


def test_sequence_measurement_rejects_invalid_denoiser_output_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before publishing evidence when the model violates tensor shape."""

    _replace_cuda_measurement(monkeypatch, peak_bytes=1)

    with pytest.raises(TypeError, match="invalid tensor shape"):
        execute_call_sequence(
            _profile(_DiffusionModel(invalid_shape=True)),
            latent=torch.zeros((2, 1, 2)),
            context=torch.zeros((2, 1, 1)),
            sample_sigmas=torch.tensor([1.0, 0.0]),
            call_count=1,
            measured=True,
        )


def _profile(diffusion: _DiffusionModel) -> PerformanceRuntimeProfile:
    """Build the smallest typed runtime profile accepted by measurement."""

    return PerformanceRuntimeProfile(
        definition=PerformanceProfile("test", 0, 0.0),
        model=_ModelPatcher(diffusion),
        model_options={"transformer_options": {}},
        executions=(),
        caches=(),
        composition=None,
    )


def _replace_cuda_measurement(
    monkeypatch: pytest.MonkeyPatch,
    *,
    peak_bytes: int,
) -> None:
    """Replace only CUDA counters while retaining real tensor execution."""

    def no_operation(device: torch.device | torch.Tensor) -> None:
        """Accept the production synchronization argument without device work."""

    def peak_memory(device: torch.device | torch.Tensor) -> int:
        """Return one deterministic peak allocation value."""

        return peak_bytes

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", no_operation)
    monkeypatch.setattr(torch.cuda, "synchronize", no_operation)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak_memory)
