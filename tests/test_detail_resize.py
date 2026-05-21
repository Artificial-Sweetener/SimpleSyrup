# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for detailer image resize routing."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.detail_resize import (
    TORCHLANC_PRECISION,
    TORCHLANC_SINC_WINDOW,
    DetailImageResizer,
)


def test_non_lanczos_upscale_uses_gpu_native_resampler() -> None:
    """Detail upscaling routes non-Lanczos methods through native GPU resize."""

    native = _FakeNativeResampler()
    lanczos = _FakeLanczosResampler()
    image = torch.ones((1, 3, 5, 2), dtype=torch.float32)

    output = DetailImageResizer(
        native_resampler_factory=lambda: native,
        lanczos_resampler_factory=lambda: lanczos,
    ).resize_up(image, 6, 8, "bicubic")

    assert output.shape == (1, 6, 8, 2)
    assert native.calls == [((1, 2, 3, 5), 8, 6, "bicubic")]
    assert lanczos.calls == []


def test_lanczos_upscale_uses_torchlanc_resampler() -> None:
    """Detail upscaling maps Lanczos to TorchLanc."""

    native = _FakeNativeResampler()
    lanczos = _FakeLanczosResampler()

    output = DetailImageResizer(
        native_resampler_factory=lambda: native,
        lanczos_resampler_factory=lambda: lanczos,
    ).resize_up(torch.ones((1, 3, 5, 2)), 7, 9, "lanczos")

    assert output.shape == (1, 7, 9, 2)
    assert native.calls == []
    assert lanczos.calls == [
        ((1, 2, 3, 5), 9, 7, TORCHLANC_SINC_WINDOW, TORCHLANC_PRECISION)
    ]


def test_downscale_always_uses_torchlanc_lanczos() -> None:
    """Detail downscaling ignores native resize and uses fixed TorchLanc settings."""

    native = _FakeNativeResampler()
    lanczos = _FakeLanczosResampler()

    output = DetailImageResizer(
        native_resampler_factory=lambda: native,
        lanczos_resampler_factory=lambda: lanczos,
    ).resize_down_lanczos(torch.ones((1, 9, 11, 3)), 4, 6)

    assert output.shape == (1, 4, 6, 3)
    assert native.calls == []
    assert lanczos.calls == [
        ((1, 3, 9, 11), 6, 4, TORCHLANC_SINC_WINDOW, TORCHLANC_PRECISION)
    ]


def test_unsupported_upscale_method_fails_clearly() -> None:
    """Unsupported detailer upscale methods fail before resize side effects."""

    with pytest.raises(ValueError, match="Unsupported sampling method"):
        DetailImageResizer().resize_up(torch.ones((1, 3, 5, 2)), 6, 8, "box")


class _FakeNativeResampler:
    """Record native resize calls and return shaped tensors."""

    def __init__(self) -> None:
        """Create empty call storage."""

        self.calls: list[tuple[tuple[int, ...], int, int, str]] = []

    def resize(
        self,
        samples: torch.Tensor,
        width: int,
        height: int,
        sampling: str,
    ) -> torch.Tensor:
        """Record the call and return a resized placeholder."""

        self.calls.append(
            (tuple(int(dim) for dim in samples.shape), width, height, sampling)
        )
        return torch.full(
            (int(samples.shape[0]), int(samples.shape[1]), height, width),
            0.25,
            dtype=samples.dtype,
        )


class _FakeLanczosResampler:
    """Record TorchLanc resize calls and return shaped tensors."""

    def __init__(self) -> None:
        """Create empty call storage."""

        self.calls: list[tuple[tuple[int, ...], int, int, int, str]] = []

    def resize(
        self,
        samples: torch.Tensor,
        width: int,
        height: int,
        sinc_window: int,
        precision: str,
    ) -> torch.Tensor:
        """Record the call and return a resized placeholder."""

        self.calls.append(
            (
                tuple(int(dim) for dim in samples.shape),
                width,
                height,
                sinc_window,
                precision,
            )
        )
        return torch.full(
            (int(samples.shape[0]), int(samples.shape[1]), height, width),
            0.75,
            dtype=samples.dtype,
        )
