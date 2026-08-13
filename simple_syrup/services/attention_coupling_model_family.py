# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the application boundary for one Attention Coupling model family."""

from __future__ import annotations

from typing import Protocol

import torch

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..runtime.attention_coupling.context_validation import RegionalContextValidator
from ..runtime.attention_coupling.family_admission import (
    AttentionCouplingFamilyAdmission,
)
from ..runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation


class AttentionCouplingModelFamily(Protocol):
    """Adapt shared preparation state to one admitted model-family backend."""

    @property
    def context_validator(self) -> RegionalContextValidator:
        """Return the family-owned processed-context validator."""

        ...

    def validate_latent(self, samples: torch.Tensor) -> None:
        """Reject latent geometry unsupported by this family."""

        ...

    def admit_adaptation(
        self,
        model: object,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> AttentionCouplingFamilyAdmission:
        """Return complete family evidence before model loading or mutation."""

        ...

    def derive(
        self,
        *,
        model: object,
        processed_plan: ProcessedRegionalAttentionPlan,
        admission: AttentionCouplingFamilyAdmission,
        region_strengths: tuple[float, ...],
        latent_batch_size: int,
    ) -> object:
        """Return one collision-safe derived model for the admitted family."""

        ...
