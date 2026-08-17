# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Profile exact model-family boundaries behind a transparent decorator."""

from __future__ import annotations

import torch

from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.raw_regional_attention import RawRegionalAttentionPlan
from simple_syrup.domain.regional_model_capabilities import RegionalModelCapabilities
from simple_syrup.runtime.attention_coupling.context_validation import (
    RegionalContextValidator,
)
from simple_syrup.runtime.attention_coupling.family_admission import (
    AttentionCouplingFamilyAdmission,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropReport,
)
from simple_syrup.services.attention_coupling_model_family import (
    AttentionCouplingModelFamily,
    AttentionCouplingPreparedModelReuse,
    AttentionCouplingSamplerConditioning,
)
from simple_syrup.services.attention_coupling_model_family_selector import (
    AttentionCouplingModelFamilySelector,
)

from .synchronized_phase_timing import measure_synchronized_phase, model_device


class ProfiledAttentionCouplingModelFamily:
    """Time exact family calls while preserving the selected family instance."""

    def __init__(self, delegate: AttentionCouplingModelFamily) -> None:
        """Retain the production-selected family as the sole behavior owner."""

        self._delegate = delegate

    @property
    def delegate(self) -> AttentionCouplingModelFamily:
        """Expose the exact wrapped family for benchmark verification."""

        return self._delegate

    @property
    def context_validator(self) -> RegionalContextValidator:
        """Return the production family's exact context validator."""

        return self._delegate.context_validator

    @property
    def prepared_model_reuse(self) -> AttentionCouplingPreparedModelReuse:
        """Return the production family's exact prepared-result policy."""

        return self._delegate.prepared_model_reuse

    def validate_latent(self, samples: torch.Tensor) -> None:
        """Delegate latent validation while timing the family boundary."""

        with measure_synchronized_phase(
            "family_validate_latent",
            device=samples.device,
        ):
            return self._delegate.validate_latent(samples)

    def admit_adaptation(
        self,
        model: object,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> AttentionCouplingFamilyAdmission:
        """Delegate admission while timing the exact selected family."""

        with measure_synchronized_phase(
            "family_admit_adaptation",
            device=model_device(model),
        ):
            return self._delegate.admit_adaptation(model, adaptation)

    def prepare_sampler_conditioning(
        self,
        plan: RawRegionalAttentionPlan,
        region_strengths: tuple[float, ...],
    ) -> AttentionCouplingSamplerConditioning:
        """Delegate sampler conditioning while timing the family boundary."""

        with measure_synchronized_phase(
            "family_sampler_conditioning",
            device=None,
        ):
            return self._delegate.prepare_sampler_conditioning(
                plan,
                region_strengths,
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
        """Delegate complete derivation with synchronized model-device timing."""

        with measure_synchronized_phase(
            "family_derive_total",
            device=model_device(model),
        ):
            return self._delegate.derive(
                model=model,
                processed_plan=processed_plan,
                admission=admission,
                interop_report=interop_report,
                region_strengths=region_strengths,
                latent_batch_size=latent_batch_size,
            )


class ProfiledAttentionCouplingModelFamilySelector(
    AttentionCouplingModelFamilySelector
):
    """Decorate the exact family selected by production capability routing."""

    def select(
        self,
        capabilities: RegionalModelCapabilities,
    ) -> AttentionCouplingModelFamily:
        """Return a transparent profiler around the production-selected family."""

        return ProfiledAttentionCouplingModelFamily(super().select(capabilities))
