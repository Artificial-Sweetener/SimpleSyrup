# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select model-family reliability calibration for concept evidence."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.regional_model_capabilities import RegionalModelFamily


@dataclass(frozen=True, slots=True)
class AttentionReliabilityCalibration:
    """Hold lightweight family-specific observation weighting constants."""

    lift_scale: float
    agreement_power: float
    consensus_influence: float
    detail_influence: float


class AttentionRegionModelReliability:
    """Own architecture-specific concept-evidence calibration selection."""

    def for_family(
        self,
        family: RegionalModelFamily,
    ) -> AttentionReliabilityCalibration:
        """Return calibrated weighting without introducing model inference."""

        if family is RegionalModelFamily.ANIMA:
            return AttentionReliabilityCalibration(0.75, 2.0, 0.75, 0.25)
        return AttentionReliabilityCalibration(1.0, 2.0, 0.8, 0.25)


ATTENTION_REGION_MODEL_RELIABILITY = AttentionRegionModelReliability()
