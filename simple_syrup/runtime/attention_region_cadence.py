# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Track denoising steps from the repeating attention-layer sequence."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Protocol

import torch

from .denoising_progress import DENOISING_PROGRESS_RESOLVER
from .regional_attention_model_call_values import uniform_model_call_sigma


class ProgressResolver(Protocol):
    """Resolve normalized progress from one exact sigma schedule."""

    def resolve(self, sample_sigmas: torch.Tensor, current_sigma: object) -> float:
        """Return normalized progress for one model call."""


class AttentionCaptureCadence:
    """Rotate captured layers and synchronize sigma once per selected step."""

    def __init__(
        self,
        stride: int,
        resolver: ProgressResolver = DENOISING_PROGRESS_RESOLVER,
    ) -> None:
        """Validate collaborators and initialize an empty layer cycle."""

        if type(stride) is not int or stride < 1:
            raise ValueError("Attention capture stride must be positive.")
        if not callable(getattr(resolver, "resolve", None)):
            raise TypeError("Attention progress resolver must provide resolve().")
        self._stride = stride
        self._resolver = resolver
        self._first_layer_key: str | None = None
        self._step_index = -1
        self._call_index = 0
        self._progress_step_index = -1
        self._progress_value: float | None = None
        self._lock = RLock()

    def sample(self, layer_key: str) -> int | None:
        """Return the current step when this rotating layer offset is selected."""

        if not layer_key:
            raise ValueError("Attention capture layer key cannot be empty.")
        with self._lock:
            if self._first_layer_key is None:
                self._first_layer_key = layer_key
                self._step_index = 0
                self._call_index = 0
            elif layer_key == self._first_layer_key:
                self._step_index += 1
                self._call_index = 0
            call_index = self._call_index
            self._call_index += 1
            phase = self._step_index % self._stride
            return self._step_index if call_index % self._stride == phase else None

    def progress(
        self,
        step_index: int,
        options: Mapping[str, object],
    ) -> float:
        """Resolve exact progress once for all selected layers in one step."""

        with self._lock:
            if (
                step_index == self._progress_step_index
                and self._progress_value is not None
            ):
                return self._progress_value
        sample_sigmas = options.get("sample_sigmas")
        current_sigmas = options.get("sigmas")
        if not isinstance(sample_sigmas, torch.Tensor):
            raise TypeError("Attention capture requires tensor sample_sigmas.")
        if not isinstance(current_sigmas, torch.Tensor):
            raise TypeError("Attention capture requires tensor sigmas.")
        progress = self._resolver.resolve(
            sample_sigmas,
            uniform_model_call_sigma(current_sigmas),
        )
        with self._lock:
            self._progress_step_index = step_index
            self._progress_value = progress
        return progress
