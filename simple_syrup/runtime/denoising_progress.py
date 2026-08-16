# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve normalized progress from an exact Comfy sampling schedule."""

from __future__ import annotations

import math

import torch


class DenoisingProgressResolver:
    """Interpolate one current sigma over a validated descending schedule."""

    def resolve(self, sample_sigmas: torch.Tensor, current_sigma: object) -> float:
        """Return normalized denoising progress in the inclusive range [0, 1]."""

        values = self._validated_sigmas(sample_sigmas)
        sigma = self._finite_scalar(current_sigma)
        steps = len(values) - 1
        if sigma >= values[0]:
            return 0.0
        if sigma <= values[-1]:
            return 1.0
        for index, (start, end) in enumerate(zip(values, values[1:], strict=True)):
            if start >= sigma >= end:
                span = start - end
                fraction = 0.0 if span == 0.0 else (start - sigma) / span
                return (index + fraction) / steps
        raise ValueError("Current sigma lies outside its sampling schedule.")

    @staticmethod
    def _validated_sigmas(sample_sigmas: torch.Tensor) -> tuple[float, ...]:
        """Return one finite descending schedule with a terminal sample value."""

        if (
            not isinstance(sample_sigmas, torch.Tensor)
            or not sample_sigmas.is_floating_point()
            or sample_sigmas.ndim != 1
            or sample_sigmas.numel() < 2
        ):
            raise ValueError(
                "Denoising sample_sigmas must be a floating vector with at least "
                "two values."
            )
        values = tuple(float(value) for value in sample_sigmas.detach().cpu().tolist())
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Denoising sample_sigmas must be finite.")
        if any(values[index] < values[index + 1] for index in range(len(values) - 1)):
            raise ValueError("Denoising sample_sigmas must be descending.")
        return values

    @staticmethod
    def _finite_scalar(value: object) -> float:
        """Normalize one scalar tensor or real current sigma."""

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Current sigma must be scalar.")
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("Current sigma must be a real number.")
        sigma = float(value)
        if not math.isfinite(sigma):
            raise ValueError("Current sigma must be finite.")
        return sigma


DENOISING_PROGRESS_RESOLVER = DenoisingProgressResolver()
