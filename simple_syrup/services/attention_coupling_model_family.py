# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the application boundary for one Attention Coupling model family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..domain.raw_regional_attention import RawRegionalAttentionPlan
from ..runtime.attention_coupling.context_validation import RegionalContextValidator
from ..runtime.attention_coupling.family_admission import (
    AttentionCouplingFamilyAdmission,
)
from ..runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from ..runtime.regional_model_patch_interop import RegionalModelPatchInteropReport


@dataclass(frozen=True, slots=True)
class AttentionCouplingSamplerConditioning:
    """Retain one family-owned positive and negative sampler pair."""

    positive: object
    negative: object


class AttentionCouplingPreparedModelReuse(StrEnum):
    """Declare whether one family admits exact prepared-request reuse."""

    DISABLED = "disabled"
    EXACT_REQUEST = "exact-request"


class AttentionCouplingModelFamily(Protocol):
    """Adapt shared preparation state to one admitted model-family backend."""

    @property
    def context_validator(self) -> RegionalContextValidator:
        """Return the family-owned processed-context validator."""

        ...

    @property
    def prepared_model_reuse(self) -> AttentionCouplingPreparedModelReuse:
        """Return the family's exact prepared-result reuse policy."""

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

    def prepare_sampler_conditioning(
        self,
        plan: RawRegionalAttentionPlan,
        region_strengths: tuple[float, ...],
    ) -> AttentionCouplingSamplerConditioning:
        """Return family-native sampler conditioning for the raw plan."""

        ...

    def derive(
        self,
        *,
        model: object,
        processed_plan: ProcessedRegionalAttentionPlan,
        admission: AttentionCouplingFamilyAdmission,
        interop_report: RegionalModelPatchInteropReport,
        region_strengths: tuple[float, ...],
        latent_batch_size: int,
    ) -> object:
        """Return one collision-safe derived model for the admitted family."""

        ...
