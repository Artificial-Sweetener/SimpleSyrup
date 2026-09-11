# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt shared Attention Coupling preparation to the standard-UNet backend."""

from __future__ import annotations

from typing import ClassVar

import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..domain.raw_regional_attention import RawRegionalAttentionPlan
from ..runtime.attention_coupling.context_validation import RegionalContextValidator
from ..runtime.attention_coupling.family_admission import (
    AttentionCouplingFamilyAdmission,
)
from ..runtime.attention_coupling.unet import StandardUnetAttentionBackend
from ..runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from ..runtime.attention_coupling.unet_context import (
    STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR,
)
from ..runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)
from ..runtime.regional_lora.standard_unet_native_admission import (
    StandardUnetNativeLoraAdmission,
    StandardUnetNativeLoraAdmissionService,
)
from ..runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from ..runtime.regional_model_patch_interop import RegionalModelPatchInteropReport
from .attention_coupling_model_family import (
    AttentionCouplingPreparedModelReuse,
    AttentionCouplingSamplerConditioning,
)

_STANDARD_UNET_BACKEND_IDENTITY = f"{UNetModel.__module__}.{UNetModel.__qualname__}"


class StandardUnetAttentionCouplingModelFamily:
    """Own standard-UNet latent, adapter, state, and derivation policy."""

    backend_class: ClassVar[type[StandardUnetAttentionBackend]] = (
        StandardUnetAttentionBackend
    )
    native_admission_class: ClassVar[type[StandardUnetNativeLoraAdmissionService]] = (
        StandardUnetNativeLoraAdmissionService
    )

    @property
    def prepared_model_reuse(self) -> AttentionCouplingPreparedModelReuse:
        """Reuse only exact standard-UNet prepared requests across sampler seeds."""

        return AttentionCouplingPreparedModelReuse.EXACT_REQUEST

    @property
    def context_validator(self) -> RegionalContextValidator:
        """Return the variable-length standard-UNet context policy."""

        return STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR

    def validate_latent(self, samples: torch.Tensor) -> None:
        """Require ordinary B/C/H/W image latents."""

        if not isinstance(samples, torch.Tensor):
            raise TypeError("Standard UNet Attention Coupling requires latent samples.")
        if samples.ndim != 4:
            raise ValueError(
                "Standard UNet Attention Coupling requires a BxCxHxW image latent."
            )

    def admit_adaptation(
        self,
        model: object,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> AttentionCouplingFamilyAdmission:
        """Resolve the complete conventional regional adapter surface."""

        if not isinstance(adaptation, RegionalLoraPlanAdaptation):
            raise TypeError(
                "Standard UNet Attention Coupling requires regional adaptation."
            )
        return self.native_admission_class().admit(model, adaptation)

    def prepare_sampler_conditioning(
        self,
        plan: RawRegionalAttentionPlan,
        region_strengths: tuple[float, ...],
    ) -> AttentionCouplingSamplerConditioning:
        """Preserve base conditioning for packed operation execution."""

        if not isinstance(plan, RawRegionalAttentionPlan):
            raise TypeError("Standard UNet sampler conditioning requires a raw plan.")
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
        """Build shared diagnostics state and derive the paired attn2 backend."""

        if not isinstance(admission, StandardUnetNativeLoraAdmission):
            raise TypeError("Standard UNet derivation requires native admission.")
        if not isinstance(interop_report, RegionalModelPatchInteropReport):
            raise TypeError("Standard UNet derivation requires interop evidence.")
        if admission.adaptation.plan != processed_plan.lora_plan:
            raise ValueError(
                "Standard UNet admission and processed conditioning must share "
                "the same regional LoRA plan."
            )
        if isinstance(latent_batch_size, bool) or not isinstance(
            latent_batch_size, int
        ):
            raise TypeError("Standard UNet latent batch size must be an integer.")
        if latent_batch_size < 1:
            raise ValueError("Standard UNet latent batch size must be positive.")
        state = StandardUnetAttentionState(
            processed_plan,
            region_strengths,
            RegionalAttentionDiagnosticsBuilder(
                processed_plan.mask_bank,
                backend=_STANDARD_UNET_BACKEND_IDENTITY,
            ),
        )
        return (
            self.backend_class()
            .derive(
                model=model,
                state=state,
                admission=admission,
                negpip=interop_report.negpip,
            )
            .model
        )
