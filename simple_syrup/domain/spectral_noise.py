"""Lift separated residuals in orthonormal spatial frequency coordinates."""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import torch


@lru_cache(maxsize=16)
def dct_basis(length: int) -> torch.Tensor:
    """Build a CPU DCT-II basis without retaining accelerator allocations."""
    if length < 1:
        raise ValueError("DCT dimensions must be positive.")
    frequency = torch.arange(length, dtype=torch.float64)[:, None]
    position = torch.arange(length, dtype=torch.float64)[None, :] + 0.5
    matrix = torch.cos(torch.pi * frequency * position / length) * (2 / length) ** 0.5
    matrix[0] /= 2**0.5
    return cast(torch.Tensor, matrix.float())


def spatial_dct(value: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Transform spatial axes while preserving batch, channel and frame axes."""
    result = value.float()
    for axis in (-1, -2):
        matrix = dct_basis(result.shape[axis]).to(result.device)
        result = result.movedim(axis, -1)
        result = result @ (matrix if inverse else matrix.T)
        result = result.movedim(-1, axis)
    return result


def lift_residual(residual: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
    """Retain source coefficients and fill only newly introduced frequency bands."""
    if (
        residual.ndim not in (4, 5)
        or residual.ndim != fresh.ndim
        or residual.shape[:-2] != fresh.shape[:-2]
    ):
        raise ValueError(
            "Noise reuse requires matching batch, channel and frame dimensions."
        )
    if any(a > b for a, b in zip(residual.shape[-2:], fresh.shape[-2:], strict=True)):
        raise ValueError(
            "Noise reuse requires spatial enlargement or unchanged dimensions."
        )
    if not bool(torch.isfinite(residual).all()) or not bool(
        torch.isfinite(fresh).all()
    ):
        raise ValueError(
            "Noise reuse requires finite residual and fresh noise tensors."
        )
    height, width = residual.shape[-2:]
    old_spectrum = torch.zeros_like(fresh, dtype=torch.float32)
    old_spectrum[..., :height, :width] = spatial_dct(residual)
    new_spectrum = spatial_dct(fresh).clone()
    new_spectrum[..., :height, :width] = 0
    return spatial_dct(old_spectrum, inverse=True) + spatial_dct(
        new_spectrum, inverse=True
    )
