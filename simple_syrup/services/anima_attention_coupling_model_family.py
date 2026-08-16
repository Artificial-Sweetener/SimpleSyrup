# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt shared Attention Coupling preparation to the Anima backend."""

from __future__ import annotations

from typing import ClassVar

import torch

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..domain.raw_regional_attention import RawRegionalAttentionPlan
from ..runtime.attention_coupling.anima_context import (
    ANIMA_REGIONAL_CONTEXT_VALIDATOR,
)
from ..runtime.attention_coupling.context_validation import RegionalContextValidator
from ..runtime.attention_coupling.family_admission import (
    AttentionCouplingFamilyAdmission,
)
from ..runtime.regional_lora.anima_full_context_backend import (
    FullContextAnimaAttentionBackend,
)
from ..runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from ..runtime.regional_model_patch_interop import RegionalModelPatchInteropReport
from .attention_coupling_model_family import (
    AttentionCouplingPreparedModelReuse,
    AttentionCouplingSamplerConditioning,
)


class AnimaAttentionCouplingModelFamily:
    """Own Anima latent, context, regional-LoRA, and derivation policy."""

    backend_class: ClassVar[type[FullContextAnimaAttentionBackend]] = (
        FullContextAnimaAttentionBackend
    )

    @property
    def prepared_model_reuse(self) -> AttentionCouplingPreparedModelReuse:
        """Preserve specialized Anima derivation and lifecycle on every request."""

        return AttentionCouplingPreparedModelReuse.DISABLED

    @property
    def context_validator(self) -> RegionalContextValidator:
        """Return the exact installed Anima context policy."""

        return ANIMA_REGIONAL_CONTEXT_VALIDATOR

    def validate_latent(self, samples: torch.Tensor) -> None:
        """Require one generated temporal slice in B/C/T/H/W layout."""

        if not isinstance(samples, torch.Tensor):
            raise TypeError("Anima Attention Coupling requires latent samples.")
        if samples.ndim != 5 or int(samples.shape[2]) != 1:
            raise ValueError(
                "Anima Attention Coupling requires a BxCx1xHxW image latent."
            )

    def admit_adaptation(
        self,
        model: object,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> AttentionCouplingFamilyAdmission:
        """Admit typed regional LoRAs without changing the Anima runtime route."""

        del model
        if not isinstance(adaptation, RegionalLoraPlanAdaptation):
            raise TypeError("Anima Attention Coupling requires regional adaptation.")
        return AttentionCouplingFamilyAdmission(adaptation)

    def prepare_sampler_conditioning(
        self,
        plan: RawRegionalAttentionPlan,
        region_strengths: tuple[float, ...],
    ) -> AttentionCouplingSamplerConditioning:
        """Preserve the existing Anima base-only sampler conditioning."""

        if not isinstance(plan, RawRegionalAttentionPlan):
            raise TypeError("Anima sampler conditioning requires a raw plan.")
        del region_strengths
        return AttentionCouplingSamplerConditioning(
            plan.positive.base_conditioning,
            plan.negative.base_conditioning,
        )

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
        """Derive the existing full-surface Anima attention and LoRA backend."""

        if not isinstance(interop_report, RegionalModelPatchInteropReport):
            raise TypeError("Anima derivation requires model interop evidence.")

        built = self.backend_class().derive(
            model=model,
            processed_plan=processed_plan,
            adaptation=admission.adaptation,
            region_strengths=region_strengths,
            latent_batch_size=latent_batch_size,
        )
        return built.model
