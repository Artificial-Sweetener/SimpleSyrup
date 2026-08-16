# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own explicit global-plus-regional standard-UNet attention weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class StandardUnetAttentionWeights:
    """Retain prevalidated standard-UNet tensor weights on one query grid."""

    base: torch.Tensor
    regions: torch.Tensor
    denominator: torch.Tensor

    def __post_init__(self) -> None:
        """Require aligned floating tensor structure without device scalar reads."""

        if not all(
            isinstance(value, torch.Tensor)
            for value in (self.base, self.regions, self.denominator)
        ):
            raise TypeError("Standard UNet attention weights must be tensors.")
        if not all(
            value.is_floating_point()
            for value in (self.base, self.regions, self.denominator)
        ):
            raise TypeError("Standard UNet attention weights must be floating point.")
        if self.base.ndim < 1:
            raise ValueError("Standard UNet base weights require a query grid.")
        if self.regions.ndim != self.base.ndim + 1 or int(self.regions.shape[0]) < 1:
            raise ValueError("Standard UNet region weights require a leading region.")
        if tuple(self.regions.shape[1:]) != tuple(self.base.shape):
            raise ValueError("Standard UNet region weights must match the base grid.")
        if self.denominator.shape != self.base.shape:
            raise ValueError("Standard UNet denominator must match the base grid.")
        if not (
            self.base.dtype == self.regions.dtype == self.denominator.dtype
            and self.base.device == self.regions.device == self.denominator.device
        ):
            raise ValueError(
                "Standard UNet attention weights must share dtype and device."
            )

    @property
    def normalized_base(self) -> torch.Tensor:
        """Return normalized global-branch weights."""

        return self.base / self.denominator

    @property
    def normalized_regions(self) -> torch.Tensor:
        """Return normalized ordered regional weights."""

        return self.regions / self.denominator.unsqueeze(0)


class StandardUnetAttentionWeightingPolicy:
    """Normalize one full-canvas global branch with masked regional branches."""

    def weights(
        self,
        masks: torch.Tensor,
        *,
        region_strengths: tuple[float, ...],
        epsilon: float = 1e-6,
    ) -> StandardUnetAttentionWeights:
        """Return PPM-compatible explicit-base weights for standard UNet."""

        self._validate_inputs(masks, region_strengths, epsilon)
        strength_shape = (len(region_strengths),) + (1,) * (masks.ndim - 1)
        strengths = masks.new_tensor(region_strengths).reshape(strength_shape)
        regions = masks.clamp(0.0, 1.0) * strengths
        base_strength = max(0.0, 1.0 - max(region_strengths))
        regional_sum = regions.sum(dim=0)
        regional_support = regional_sum.ne(0)
        base = torch.where(
            regional_support,
            masks.new_full(masks.shape[1:], base_strength),
            masks.new_ones(masks.shape[1:]),
        )
        denominator = (base + regional_sum).clamp_min(epsilon)
        return StandardUnetAttentionWeights(base, regions, denominator)

    @staticmethod
    def blend(
        *,
        weights: StandardUnetAttentionWeights,
        base_output: torch.Tensor,
        regional_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Blend prevalidated outputs without synchronizing device values."""

        if not isinstance(weights, StandardUnetAttentionWeights):
            raise TypeError("Standard UNet blend requires typed weights.")
        if not isinstance(base_output, torch.Tensor) or not isinstance(
            regional_outputs, torch.Tensor
        ):
            raise TypeError("Standard UNet attention outputs must be tensors.")
        if base_output.dtype != regional_outputs.dtype:
            raise ValueError("Standard UNet attention output dtypes must match.")
        if base_output.device != regional_outputs.device:
            raise ValueError("Standard UNet attention output devices must match.")
        if regional_outputs.shape != (
            int(weights.regions.shape[0]),
            *base_output.shape,
        ):
            raise ValueError("Standard UNet regional output shape is invalid.")
        query_dimensions = weights.base.ndim
        if base_output.ndim < query_dimensions or tuple(
            base_output.shape[:query_dimensions]
        ) != tuple(weights.base.shape):
            raise ValueError("Standard UNet output query grid is invalid.")
        feature_shape = (1,) * (base_output.ndim - query_dimensions)
        base = weights.base.reshape((*weights.base.shape, *feature_shape)).to(
            dtype=base_output.dtype
        )
        regions = weights.regions.reshape((*weights.regions.shape, *feature_shape)).to(
            dtype=base_output.dtype
        )
        denominator = weights.denominator.reshape(
            (*weights.denominator.shape, *feature_shape)
        ).to(dtype=base_output.dtype)
        return (
            base * base_output + (regions * regional_outputs).sum(dim=0)
        ) / denominator

    @staticmethod
    def _validate_inputs(
        masks: object,
        region_strengths: object,
        epsilon: object,
    ) -> None:
        """Validate host and tensor structure after canonical mask admission."""

        if not isinstance(masks, torch.Tensor):
            raise TypeError("Standard UNet attention masks must be a tensor.")
        if (
            masks.ndim < 2
            or int(masks.shape[0]) < 1
            or any(int(size) < 1 for size in masks.shape[1:])
        ):
            raise ValueError("Standard UNet masks require region and query grids.")
        if not masks.is_floating_point():
            raise TypeError("Standard UNet masks must use floating point.")
        if not isinstance(region_strengths, tuple) or len(region_strengths) != int(
            masks.shape[0]
        ):
            raise ValueError("Standard UNet strength count must match the regions.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in region_strengths
        ):
            raise ValueError("Standard UNet strengths must be finite and non-negative.")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, int | float)
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
        ):
            raise ValueError("Standard UNet epsilon must be finite and positive.")


STANDARD_UNET_ATTENTION_WEIGHTING_POLICY = StandardUnetAttentionWeightingPolicy()
