# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Retain immutable model-family regional-adapter admission evidence."""

from __future__ import annotations

from dataclasses import dataclass

from ..regional_lora_plan_adapter import RegionalLoraPlanAdaptation


@dataclass(frozen=True, slots=True)
class AttentionCouplingFamilyAdmission:
    """Retain the exact adaptation admitted before model loading."""

    adaptation: RegionalLoraPlanAdaptation

    def __post_init__(self) -> None:
        """Require typed immutable adaptation evidence."""

        if not isinstance(self.adaptation, RegionalLoraPlanAdaptation):
            raise TypeError("Attention Coupling admission requires adaptation.")
