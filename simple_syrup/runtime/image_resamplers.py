# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI-native image resampling adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, cast

import torch

SUPPORTED_SAMPLERS = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")
Processor = Literal["cpu", "gpu"]
CommonUpscale = Callable[[torch.Tensor, int, int, str, str], torch.Tensor]
DeviceProvider = Callable[[], torch.device | str]


class ImageResampler(Protocol):
    """Resize BCHW image tensors to the requested size."""

    def resize(
        self,
        samples: torch.Tensor,
        width: int,
        height: int,
        sampling: str,
    ) -> torch.Tensor:
        """Return resized BCHW samples."""


class NativeComfyResampler:
    """Resize images through ComfyUI's native Pillow/PyTorch path."""

    def __init__(
        self,
        processor: Processor,
        common_upscale: CommonUpscale | None = None,
        device_provider: DeviceProvider | None = None,
    ) -> None:
        """Create a native resampler for the selected processor."""

        if processor not in ("cpu", "gpu"):
            raise ValueError(f"processor must be 'cpu' or 'gpu', got {processor!r}.")
        self._processor = processor
        self._common_upscale = common_upscale
        self._device_provider = device_provider

    def resize(
        self,
        samples: torch.Tensor,
        width: int,
        height: int,
        sampling: str,
    ) -> torch.Tensor:
        """Resize BCHW samples with ComfyUI's native implementation."""

        validate_sampling(sampling)
        device = torch.device("cpu")
        if self._processor == "gpu":
            device = self._get_torch_device()

        samples_on_device = samples.to(device=device)
        return self._get_common_upscale()(
            samples_on_device,
            int(width),
            int(height),
            sampling,
            "disabled",
        )

    def _get_common_upscale(self) -> CommonUpscale:
        """Return ComfyUI's common upscale function or an injected test double."""

        if self._common_upscale is not None:
            return self._common_upscale

        import comfy.utils

        return cast(CommonUpscale, comfy.utils.common_upscale)

    def _get_torch_device(self) -> torch.device:
        """Return ComfyUI's configured torch execution device."""

        if self._device_provider is not None:
            return torch.device(self._device_provider())

        from comfy import model_management

        return torch.device(model_management.get_torch_device())


def validate_sampling(sampling: str) -> None:
    """Validate a ComfyUI image resize sampler name."""

    if sampling not in SUPPORTED_SAMPLERS:
        supported = ", ".join(SUPPORTED_SAMPLERS)
        raise ValueError(
            f"Unsupported sampling method {sampling!r}. Use one of: {supported}."
        )
