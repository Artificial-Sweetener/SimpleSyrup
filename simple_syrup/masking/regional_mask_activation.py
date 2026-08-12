# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify active regional coverage on one projected query grid."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class RegionalCoverageClass(StrEnum):
    """Identify model branches required by one regional mask projection."""

    ALL_BASE = "all_base"
    ALL_SINGLE_REGION = "all_single_region"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class RegionalMaskActivation:
    """Describe ordered active regions and an admitted coverage fast path."""

    coverage_class: RegionalCoverageClass
    active_region_indices: tuple[int, ...]
    single_region_index: int | None

    def __post_init__(self) -> None:
        """Require one internally consistent immutable classification."""

        if not isinstance(self.coverage_class, RegionalCoverageClass):
            raise TypeError(
                "Regional coverage class must be a RegionalCoverageClass value."
            )
        if not isinstance(self.active_region_indices, tuple):
            raise TypeError("Active regional indices must be an immutable tuple.")
        if any(index < 0 for index in self.active_region_indices):
            raise ValueError("Active regional indices must be non-negative.")
        if tuple(sorted(set(self.active_region_indices))) != self.active_region_indices:
            raise ValueError("Active regional indices must be unique and ordered.")
        if self.coverage_class is RegionalCoverageClass.ALL_BASE:
            if self.active_region_indices or self.single_region_index is not None:
                raise ValueError("All-base coverage cannot contain an active region.")
            return
        if self.coverage_class is RegionalCoverageClass.ALL_SINGLE_REGION:
            if (
                len(self.active_region_indices) != 1
                or self.single_region_index != self.active_region_indices[0]
            ):
                raise ValueError(
                    "All-single-region coverage requires its one active region index."
                )
            return
        if not self.active_region_indices:
            raise ValueError("Mixed regional coverage requires an active region.")
        if self.single_region_index is not None:
            raise ValueError(
                "Mixed regional coverage cannot select a fast-path region."
            )


class RegionalMaskActivationClassifier:
    """Own zero-coverage pruning and regional fast-path admission."""

    def classify(
        self,
        masks: torch.Tensor,
        *,
        tolerance: float = 1e-6,
    ) -> RegionalMaskActivation:
        """Return ordered active regions and the exact coverage classification."""

        self._validate_masks(masks)
        if not 0.0 <= tolerance < 0.5:
            raise ValueError(
                "Regional activation tolerance must be at least 0 and below 0.5."
            )
        active_region_indices = tuple(
            index
            for index in range(int(masks.shape[0]))
            if bool((masks[index] > tolerance).any())
        )
        if not active_region_indices:
            return RegionalMaskActivation(
                coverage_class=RegionalCoverageClass.ALL_BASE,
                active_region_indices=(),
                single_region_index=None,
            )
        if len(active_region_indices) == 1:
            region_index = active_region_indices[0]
            if bool((masks[region_index] >= 1.0 - tolerance).all()):
                return RegionalMaskActivation(
                    coverage_class=RegionalCoverageClass.ALL_SINGLE_REGION,
                    active_region_indices=active_region_indices,
                    single_region_index=region_index,
                )
        return RegionalMaskActivation(
            coverage_class=RegionalCoverageClass.MIXED,
            active_region_indices=active_region_indices,
            single_region_index=None,
        )

    @staticmethod
    def _validate_masks(masks: torch.Tensor) -> None:
        """Validate a projected normalized floating-point BHW mask batch."""

        if not isinstance(masks, torch.Tensor):
            raise TypeError("Regional activation masks must be a torch.Tensor.")
        if masks.ndim != 3:
            raise ValueError("Regional activation masks must use BHW layout.")
        if int(masks.shape[0]) < 1:
            raise ValueError("Regional activation requires at least one region.")
        if int(masks.shape[1]) < 1 or int(masks.shape[2]) < 1:
            raise ValueError("Regional activation mask grids must be non-empty.")
        if not masks.is_floating_point():
            raise TypeError(
                "Regional activation masks must use a floating-point dtype."
            )
        if not bool(torch.isfinite(masks).all()):
            raise ValueError("Regional activation masks must contain finite values.")
        if not bool(((masks >= 0.0) & (masks <= 1.0)).all()):
            raise ValueError("Regional activation masks must stay within [0, 1].")
