# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Describe native trajectory captures and independent finishing controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .noise_inversion import InversionMethod

PROVENANCE_KEY = "simple_syrup_sampling_provenance"


@dataclass(frozen=True, slots=True)
class NoiseReuseOptions:
    """Request source noise at the receiving schedule's starting coordinate."""

    finishing_steps: int = 0
    finishing_method: InversionMethod = "euler"
    finishing_start_fraction: float = 0.75

    def __post_init__(self) -> None:
        """Reject singular coordinates and unbounded integration requests."""
        if (
            type(self.finishing_steps) is not int
            or not 0 <= self.finishing_steps <= 100
        ):
            raise ValueError("Finishing steps must be between 0 and 100.")
        if self.finishing_method not in ("euler", "heun"):
            raise ValueError("Finishing method must be Euler or Heun.")
        if (
            not math.isfinite(self.finishing_start_fraction)
            or not 0 < self.finishing_start_fraction <= 1
        ):
            raise ValueError(
                "Finishing start fraction must be greater than 0 and at most 1."
            )
        if self.finishing_steps and self.finishing_start_fraction == 1:
            raise ValueError(
                "Finishing start fraction must be below 1 when finishing is enabled."
            )

    def capture_coordinate(self, target: float) -> float:
        """Round the requested coordinate exactly as the native float32 schedule."""
        fraction = self.finishing_start_fraction if self.finishing_steps else 1.0
        return float(torch.tensor(target * fraction, dtype=torch.float32))


@dataclass(frozen=True, slots=True)
class CapturedSamplingState:
    """Associate a requested level with a real observed state and recovered noise."""

    requested_coordinate: float
    actual_coordinate: float
    state: torch.Tensor
    residual: torch.Tensor

    def __post_init__(self) -> None:
        """Require finite positive coordinates and matching real tensor layouts."""
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.requested_coordinate, self.actual_coordinate)
        ):
            raise ValueError("Saved noise levels must be finite and greater than zero.")
        if self.state.shape != self.residual.shape:
            raise ValueError(
                "Saved noisy state and recovered noise must have one shape."
            )
        if not bool(torch.isfinite(self.state).all()) or not bool(
            torch.isfinite(self.residual).all()
        ):
            raise ValueError("The original sampler saved invalid noise values.")


@dataclass(frozen=True, slots=True)
class SamplingProvenance:
    """Carry the source identity and real model-space states beside external latents."""

    latent_format: str
    clean: torch.Tensor
    captures: tuple[CapturedSamplingState, ...] = ()
    schedule: torch.Tensor | None = None
    graph_context: str = ""

    def capture_at(self, coordinate: float) -> CapturedSamplingState:
        """Require an observation explicitly associated with this receiving request."""
        for capture in self.captures:
            if capture.requested_coordinate == coordinate:
                return capture
        raise ValueError(
            f"The original sampling run did not save noise for the requested level "
            f"{coordinate:.9g}. Queue the connected workflow again with Noise Reuse "
            "Options enabled."
        )

    def state_at(self, coordinate: float) -> torch.Tensor:
        """Return the real noisy state associated with the requested level."""
        return self.capture_at(coordinate).state

    def residual_at(self, coordinate: float) -> torch.Tensor:
        """Return noise separated with the originating model's own parameterization."""
        return self.capture_at(coordinate).residual
