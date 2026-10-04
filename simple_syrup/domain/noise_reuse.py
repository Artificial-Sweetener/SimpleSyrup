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
class SamplingProvenance:
    """Carry the source identity and real model-space states beside external latents."""

    latent_format: str
    clean: torch.Tensor
    states: tuple[tuple[float, torch.Tensor], ...] = ()
    schedule: torch.Tensor | None = None
    residuals: tuple[tuple[float, torch.Tensor], ...] = ()

    def state_at(self, coordinate: float) -> torch.Tensor:
        """Require an exact requested schedule coordinate instead of approximating."""
        for sigma, state in self.states:
            if sigma == coordinate:
                return state
        raise ValueError(
            f"Source sampling did not capture sigma {coordinate:.9g}. "
            "Queue the connected workflow again with Noise Reuse Options enabled."
        )

    def residual_at(self, coordinate: float) -> torch.Tensor:
        """Return noise separated with the originating model's own parameterization."""
        self.state_at(coordinate)
        for sigma, residual in self.residuals:
            if sigma == coordinate:
                return residual
        raise ValueError("Source capture is missing its native recovered residual.")
