# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preserve global composition inside active Anima attention regions."""

from __future__ import annotations

import math

import torch

from ...domain.regional_attention_weights import (
    RegionalAttentionWeightingPolicy,
    RegionalAttentionWeights,
)

ANIMA_GLOBAL_ATTENTION_SHARE = 1.0 / 3.0


class AnimaCrossAttentionWeightingPolicy(RegionalAttentionWeightingPolicy):
    """Reserve global prompt influence inside every active region."""

    def __init__(
        self,
        global_share: float = ANIMA_GLOBAL_ATTENTION_SHARE,
    ) -> None:
        """Set the bounded global share retained inside active regions."""

        if (
            isinstance(global_share, bool)
            or not isinstance(global_share, int | float)
            or not math.isfinite(float(global_share))
            or not 0.0 < float(global_share) < 1.0
        ):
            raise ValueError(
                "Anima global attention share must be between zero and one."
            )
        self._global_share = float(global_share)

    def weights(
        self,
        masks: torch.Tensor,
        *,
        region_strengths: tuple[float, ...],
        epsilon: float = 1e-6,
    ) -> RegionalAttentionWeights:
        """Move a fixed regional share to the global prompt everywhere."""

        standard = super().weights(
            masks,
            region_strengths=region_strengths,
            epsilon=epsilon,
        )
        regional_sum = standard.regions.sum(dim=0)
        return RegionalAttentionWeights(
            base=standard.base + (regional_sum * self._global_share),
            regions=standard.regions * (1.0 - self._global_share),
            denominator=standard.denominator,
        )


ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY = AnimaCrossAttentionWeightingPolicy()
