# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for image resampler runtime adapters."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest
import torch

from simple_syrup.runtime.image_resamplers import (
    NativeComfyResampler,
    validate_sampling,
)
from simple_syrup.runtime.torchlanc_resampler import TorchLanczosResampler


def test_native_cpu_lanczos_calls_common_upscale() -> None:
    """CPU Lanczos uses the native ComfyUI upscale adapter path."""

    calls: list[tuple[torch.device, int, int, str, str]] = []

    def fake_common_upscale(
        samples: torch.Tensor,
        width: int,
        height: int,
        sampling: str,
        crop: str,
    ) -> torch.Tensor:
        """Record native upscale calls and return resized placeholder data."""

        calls.append((samples.device, width, height, sampling, crop))
        return torch.zeros(
            (samples.shape[0], samples.shape[1], height, width),
            dtype=samples.dtype,
            device=samples.device,
        )

    samples = torch.ones((1, 3, 4, 6), dtype=torch.float32)
    output = NativeComfyResampler(
        processor="cpu",
        common_upscale=fake_common_upscale,
    ).resize(samples, 8, 10, "lanczos")

    assert calls == [(torch.device("cpu"), 8, 10, "lanczos", "disabled")]
    assert output.shape == (1, 3, 10, 8)


def test_native_cpu_non_lanczos_calls_common_upscale() -> None:
    """CPU non-Lanczos samplers use the native ComfyUI PyTorch path."""

    calls: list[str] = []

    def fake_common_upscale(
        samples: torch.Tensor,
        width: int,
        height: int,
        sampling: str,
        crop: str,
    ) -> torch.Tensor:
        """Record the selected sampler."""

        calls.append(sampling)
        return torch.zeros((samples.shape[0], samples.shape[1], height, width))

    NativeComfyResampler(
        processor="cpu",
        common_upscale=fake_common_upscale,
    ).resize(torch.ones((1, 3, 4, 6)), 8, 10, "bicubic")

    assert calls == ["bicubic"]


def test_native_gpu_uses_device_provider() -> None:
    """GPU non-Lanczos path uses ComfyUI's torch-device provider."""

    provider_calls: list[str] = []

    def fake_device_provider() -> torch.device:
        """Record that the configured device was requested."""

        provider_calls.append("called")
        return torch.device("cpu")

    def fake_common_upscale(
        samples: torch.Tensor,
        width: int,
        height: int,
        sampling: str,
        crop: str,
    ) -> torch.Tensor:
        """Return placeholder output for the selected device."""

        return torch.zeros(
            (samples.shape[0], samples.shape[1], height, width),
            device=samples.device,
        )

    NativeComfyResampler(
        processor="gpu",
        common_upscale=fake_common_upscale,
        device_provider=fake_device_provider,
    ).resize(torch.ones((1, 3, 4, 6)), 8, 10, "area")

    assert provider_calls == ["called"]


def test_validate_sampling_rejects_unknown_sampler() -> None:
    """Unsupported sampler names are rejected before execution."""

    with pytest.raises(ValueError, match="Unsupported sampling method"):
        validate_sampling("bad")


def test_torchlanc_adapter_passes_expected_arguments() -> None:
    """TorchLanc adapter maps controls to lanczos_resize arguments."""

    calls: list[dict[str, Any]] = []

    def fake_lanczos_resize(samples: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Record TorchLanc keyword arguments and return the input."""

        calls.append({"samples": samples, **kwargs})
        return samples

    samples = torch.ones((1, 3, 4, 6), dtype=torch.float32)
    output = TorchLanczosResampler(
        lanczos_resize=fake_lanczos_resize,
        cuda_available=lambda: True,
        device_provider=lambda: torch.device("cpu"),
    ).resize(samples, 8, 10, sinc_window=4, precision="fp16")

    assert output is samples
    assert calls[0]["height"] == 10
    assert calls[0]["width"] == 8
    assert calls[0]["a"] == 4
    assert calls[0]["precision"] == "fp16"
    assert calls[0]["clamp"] is True
    assert calls[0]["chunk_size"] == 0


def test_torchlanc_adapter_imports_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TorchLanc is imported only when the adapter executes."""

    imported: list[str] = []

    def fake_import_module(name: str) -> ModuleType:
        """Provide a fake TorchLanc module and record import timing."""

        imported.append(name)
        module = ModuleType("torchlanc")

        def fake_lanczos_resize(samples: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """Return samples unchanged for lazy import testing."""

            return samples

        module.lanczos_resize = fake_lanczos_resize  # type: ignore[attr-defined]
        return module

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    adapter = TorchLanczosResampler(
        cuda_available=lambda: True,
        device_provider=lambda: torch.device("cpu"),
    )

    assert imported == []
    adapter.resize(torch.ones((1, 3, 4, 6)), 8, 10, 3, "fp32")
    assert imported == ["torchlanc"]


def test_torchlanc_adapter_missing_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing TorchLanc produces a direct installation error."""

    def fake_import_module(name: str) -> ModuleType:
        """Simulate TorchLanc import failure."""

        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    adapter = TorchLanczosResampler(
        cuda_available=lambda: True,
        device_provider=lambda: torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="TorchLanc is required"):
        adapter.resize(torch.ones((1, 3, 4, 6)), 8, 10, 3, "fp32")


def test_torchlanc_adapter_requires_cuda() -> None:
    """GPU Lanczos path fails closed when CUDA is unavailable."""

    adapter = TorchLanczosResampler(cuda_available=lambda: False)

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        adapter.resize(torch.ones((1, 3, 4, 6)), 8, 10, 3, "fp32")
