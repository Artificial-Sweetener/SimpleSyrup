# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Normalize regional attention weights and blend ordered branch outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RegionalAttentionWeights:
    """Hold raw base, region, and denominator weights on one query grid."""

    base: torch.Tensor
    regions: torch.Tensor
    denominator: torch.Tensor

    def __post_init__(self) -> None:
        """Require consistent finite non-negative weighting tensors."""

        for name, tensor in (
            ("Base", self.base),
            ("Region", self.regions),
            ("Denominator", self.denominator),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} attention weights must be a torch.Tensor.")
            if not tensor.is_floating_point():
                raise TypeError(
                    f"{name} attention weights must use a floating-point dtype."
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} attention weights must be finite.")
            if not bool((tensor >= 0.0).all()):
                raise ValueError(f"{name} attention weights must be non-negative.")
        if self.base.ndim < 1:
            raise ValueError("Base attention weights require a query-grid dimension.")
        if self.regions.ndim != self.base.ndim + 1:
            raise ValueError(
                "Region attention weights require a leading region dimension."
            )
        if int(self.regions.shape[0]) < 1:
            raise ValueError("Region attention weights require at least one region.")
        if tuple(self.regions.shape[1:]) != tuple(self.base.shape):
            raise ValueError("Region attention weights must match the base query grid.")
        if self.denominator.shape != self.base.shape:
            raise ValueError("Attention denominator must match the base query grid.")
        if not (
            self.base.dtype == self.regions.dtype == self.denominator.dtype
            and self.base.device == self.regions.device == self.denominator.device
        ):
            raise ValueError(
                "Base, region, and denominator weights must share dtype and device."
            )
        if not bool((self.denominator > 0.0).all()):
            raise ValueError("Attention denominator must be strictly positive.")

    @property
    def normalized_base(self) -> torch.Tensor:
        """Return normalized base-complement weights."""

        return self.base / self.denominator

    @property
    def normalized_regions(self) -> torch.Tensor:
        """Return normalized ordered regional weights."""

        return self.regions / self.denominator.unsqueeze(0)


class RegionalAttentionWeightingPolicy:
    """Own Comfy-compatible base complement and normalized branch composition."""

    def weights(
        self,
        masks: torch.Tensor,
        *,
        region_strengths: tuple[float, ...],
        epsilon: float = 1e-6,
    ) -> RegionalAttentionWeights:
        """Return raw weighting terms for ordered region-first query masks."""

        self._validate_mask_structure(masks)
        strengths = self._validate_strengths(
            region_strengths,
            region_count=int(masks.shape[0]),
        )
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, int | float)
            or not math.isfinite(float(epsilon))
            or epsilon <= 0.0
        ):
            raise ValueError("Regional attention epsilon must be finite and positive.")
        self._validate_mask_values(masks)

        strength_shape = (len(strengths),) + (1,) * (masks.ndim - 1)
        strength_tensor = masks.new_tensor(strengths).reshape(strength_shape)
        region_weights = masks.clamp(0.0, 1.0) * strength_tensor
        region_sum = region_weights.sum(dim=0)
        base_weight = torch.relu(1.0 - region_sum)
        denominator = (base_weight + region_sum).clamp_min(float(epsilon))
        return RegionalAttentionWeights(
            base=base_weight,
            regions=region_weights,
            denominator=denominator,
        )

    def blend(
        self,
        *,
        weights: RegionalAttentionWeights,
        base_output: torch.Tensor,
        regional_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Blend one base and ordered regional outputs over their query grid."""

        self._validate_outputs(
            weights=weights,
            base_output=base_output,
            regional_outputs=regional_outputs,
        )
        feature_dimensions = base_output.ndim - weights.base.ndim
        feature_shape = (1,) * feature_dimensions
        base_weight = weights.base.reshape((*weights.base.shape, *feature_shape)).to(
            dtype=base_output.dtype
        )
        region_weights = weights.regions.reshape(
            (*weights.regions.shape, *feature_shape)
        ).to(dtype=base_output.dtype)
        denominator = weights.denominator.reshape(
            (*weights.denominator.shape, *feature_shape)
        ).to(dtype=base_output.dtype)
        numerator = base_weight * base_output + (region_weights * regional_outputs).sum(
            dim=0
        )
        blended = numerator / denominator
        if not bool(torch.isfinite(blended).all()):
            raise ValueError(
                "Blended regional attention output contains non-finite values."
            )
        return blended

    @staticmethod
    def _validate_mask_structure(masks: torch.Tensor) -> None:
        """Validate mask type and shape without inspecting device values."""

        if not isinstance(masks, torch.Tensor):
            raise TypeError("Regional attention masks must be a torch.Tensor.")
        if masks.ndim < 2:
            raise ValueError(
                "Regional attention masks require region and query-grid dimensions."
            )
        if int(masks.shape[0]) < 1 or any(int(size) < 1 for size in masks.shape[1:]):
            raise ValueError(
                "Regional attention masks require non-empty region and query grids."
            )
        if not masks.is_floating_point():
            raise TypeError("Regional attention masks must use a floating-point dtype.")

    @staticmethod
    def _validate_mask_values(masks: torch.Tensor) -> None:
        """Reject non-finite mask values before constructing strength tensors."""

        if not bool(torch.isfinite(masks).all()):
            raise ValueError("Regional attention masks must contain finite values.")

    @staticmethod
    def _validate_strengths(
        strengths: tuple[float, ...],
        *,
        region_count: int,
    ) -> tuple[float, ...]:
        """Validate ordered immutable regional strengths before tensor creation."""

        if not isinstance(strengths, tuple):
            raise TypeError("Regional attention strengths must be an immutable tuple.")
        if len(strengths) != region_count:
            raise ValueError(
                "Regional attention strength count must match the mask region count."
            )
        normalized: list[float] = []
        for index, strength in enumerate(strengths):
            if isinstance(strength, bool) or not isinstance(strength, int | float):
                raise TypeError(
                    f"Regional attention strength {index} must be a real number."
                )
            value = float(strength)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Regional attention strength {index} must be finite and "
                    "non-negative."
                )
            normalized.append(value)
        return tuple(normalized)

    @staticmethod
    def _validate_outputs(
        *,
        weights: RegionalAttentionWeights,
        base_output: torch.Tensor,
        regional_outputs: torch.Tensor,
    ) -> None:
        """Validate branch output ordering, grids, features, and tensor state."""

        if not isinstance(weights, RegionalAttentionWeights):
            raise TypeError("Regional blend weights must be RegionalAttentionWeights.")
        if not isinstance(base_output, torch.Tensor) or not isinstance(
            regional_outputs, torch.Tensor
        ):
            raise TypeError(
                "Regional attention branch outputs must be torch.Tensor values."
            )
        if (
            not base_output.is_floating_point()
            or not regional_outputs.is_floating_point()
        ):
            raise TypeError(
                "Regional attention branch outputs must use floating-point dtypes."
            )
        if base_output.dtype != regional_outputs.dtype:
            raise ValueError("Regional attention branch output dtypes must match.")
        if base_output.device != regional_outputs.device:
            raise ValueError("Regional attention branch output devices must match.")
        expected_regional_shape = (int(weights.regions.shape[0]), *base_output.shape)
        if regional_outputs.shape != expected_regional_shape:
            raise ValueError(
                "Regional outputs must contain one ordered branch per region with "
                "the complete base output shape."
            )
        query_dimensions = weights.base.ndim
        if base_output.ndim < query_dimensions or tuple(
            base_output.shape[:query_dimensions]
        ) != tuple(weights.base.shape):
            raise ValueError(
                "Regional attention outputs must begin with the weighting query grid."
            )
        if weights.base.device != base_output.device:
            raise ValueError(
                "Regional weights and branch outputs must share one device."
            )
        if not bool(torch.isfinite(base_output).all()) or not bool(
            torch.isfinite(regional_outputs).all()
        ):
            raise ValueError(
                "Regional attention branch outputs must contain finite values."
            )
