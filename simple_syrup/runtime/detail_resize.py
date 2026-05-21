# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Runtime image resizing policy for detailer scaling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from .image_resamplers import NativeComfyResampler, validate_sampling
from .torchlanc_resampler import TorchLanczosResampler

SUPPORTED_DETAIL_UPSCALE_METHODS = (
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos",
)
TORCHLANC_SINC_WINDOW = 3
TORCHLANC_PRECISION = "fp32"


class DetailNativeResampler(Protocol):
    """Resize BCHW tensors with a selected non-Lanczos method."""

    def resize(
        self,
        samples: torch.Tensor,
        width: int,
        height: int,
        sampling: str,
    ) -> torch.Tensor:
        """Return resized BCHW samples."""


class DetailLanczosResampler(Protocol):
    """Resize BCHW tensors with TorchLanc Lanczos."""

    def resize(
        self,
        samples: torch.Tensor,
        width: int,
        height: int,
        sinc_window: int,
        precision: str,
    ) -> torch.Tensor:
        """Return resized BCHW samples."""


NativeResamplerFactory = Callable[[], DetailNativeResampler]
LanczosResamplerFactory = Callable[[], DetailLanczosResampler]


class DetailImageResizer:
    """Resize detailer image tensors with the project detailer policy."""

    def __init__(
        self,
        native_resampler_factory: NativeResamplerFactory | None = None,
        lanczos_resampler_factory: LanczosResamplerFactory | None = None,
    ) -> None:
        """Create the resizer with injectable runtime adapters."""

        self._native_resampler_factory = native_resampler_factory
        self._lanczos_resampler_factory = lanczos_resampler_factory

    def resize_up(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
        method: str,
    ) -> torch.Tensor:
        """Resize a BHWC image upward using the requested detailer method."""

        self.validate_method(method)
        return (
            self._resize_lanczos(image, height, width)
            if method == "lanczos"
            else (self._resize_native(image, height, width, method))
        )

    def resize_down_lanczos(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Resize a BHWC image downward using fixed TorchLanc Lanczos."""

        return self._resize_lanczos(image, height, width)

    def validate_method(self, method: str) -> None:
        """Reject unsupported detailer upscale methods."""

        validate_sampling(method)
        if method not in SUPPORTED_DETAIL_UPSCALE_METHODS:
            supported = ", ".join(SUPPORTED_DETAIL_UPSCALE_METHODS)
            raise ValueError(
                f"Unsupported detailer upscale method {method!r}. Use one of: "
                f"{supported}."
            )

    def _resize_native(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
        method: str,
    ) -> torch.Tensor:
        """Resize a BHWC image through ComfyUI's GPU resize path."""

        samples = _bhwc_to_bchw(image)
        resized = self._native_resampler().resize(samples, width, height, method)
        return _bchw_to_bhwc(resized)

    def _resize_lanczos(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Resize a BHWC image through TorchLanc Lanczos."""

        samples = _bhwc_to_bchw(image)
        resized = self._lanczos_resampler().resize(
            samples,
            width,
            height,
            TORCHLANC_SINC_WINDOW,
            TORCHLANC_PRECISION,
        )
        return _bchw_to_bhwc(resized)

    def _native_resampler(self) -> DetailNativeResampler:
        """Return the injected or default GPU native resampler."""

        if self._native_resampler_factory is not None:
            return self._native_resampler_factory()
        return NativeComfyResampler("gpu")

    def _lanczos_resampler(self) -> DetailLanczosResampler:
        """Return the injected or default TorchLanc resampler."""

        if self._lanczos_resampler_factory is not None:
            return self._lanczos_resampler_factory()
        return TorchLanczosResampler()


def _bhwc_to_bchw(image: torch.Tensor) -> torch.Tensor:
    """Validate and convert a BHWC image tensor to BCHW."""

    if image.ndim != 4:
        raise ValueError("detailer image resizing requires a BHWC tensor.")
    if int(image.shape[0]) < 1 or int(image.shape[-1]) < 1:
        raise ValueError("detailer image resizing requires non-empty image tensors.")
    return image.float().clamp(0.0, 1.0).movedim(-1, 1)


def _bchw_to_bhwc(samples: torch.Tensor) -> torch.Tensor:
    """Validate and convert a BCHW image tensor to BHWC."""

    if samples.ndim != 4:
        raise ValueError("detailer resampler output must be a BCHW tensor.")
    return samples.movedim(1, -1).clamp(0.0, 1.0)
