# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""TorchLanc runtime adapter for GPU Lanczos resizing."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import cast

import torch

from .image_resamplers import DeviceProvider

TorchLancResize = Callable[..., torch.Tensor]
CudaAvailable = Callable[[], bool]


class TorchLanczosResampler:
    """Resize images with TorchLanc's GPU Lanczos implementation."""

    def __init__(
        self,
        lanczos_resize: TorchLancResize | None = None,
        cuda_available: CudaAvailable | None = None,
        device_provider: DeviceProvider | None = None,
    ) -> None:
        """Create a TorchLanc adapter with injectable runtime boundaries."""

        self._lanczos_resize = lanczos_resize
        self._cuda_available = cuda_available or torch.cuda.is_available
        self._device_provider = device_provider

    def resize(
        self,
        samples: torch.Tensor,
        width: int,
        height: int,
        sinc_window: int,
        precision: str,
    ) -> torch.Tensor:
        """Resize BCHW samples with TorchLanc on the configured torch device."""

        if not self._cuda_available():
            raise RuntimeError(
                "GPU Lanczos resizing requires a CUDA device. Select "
                "processor='cpu' for Pillow Lanczos."
            )

        device = self._get_torch_device()
        samples_on_device = samples.to(device=device)
        return self._get_lanczos_resize()(
            samples_on_device,
            height=int(height),
            width=int(width),
            a=int(sinc_window),
            precision=str(precision),
            clamp=True,
            chunk_size=0,
        )

    def _get_lanczos_resize(self) -> TorchLancResize:
        """Return TorchLanc's resize function, importing it lazily when needed."""

        if self._lanczos_resize is not None:
            return self._lanczos_resize

        try:
            module: ModuleType = importlib.import_module("torchlanc")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TorchLanc is required for GPU Lanczos resizing. Install this "
                "node pack's requirements into the ComfyUI virtual environment."
            ) from exc

        resize = module.lanczos_resize
        if not callable(resize):
            raise RuntimeError("TorchLanc does not expose callable lanczos_resize.")
        return cast(TorchLancResize, resize)

    def _get_torch_device(self) -> torch.device:
        """Return ComfyUI's configured torch execution device."""

        if self._device_provider is not None:
            return torch.device(self._device_provider())

        from comfy import model_management

        return torch.device(model_management.get_torch_device())
