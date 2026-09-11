# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Install one complete persistent-variant runtime on a derived standard MODEL."""

from __future__ import annotations

from dataclasses import dataclass

from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP, WrappersMP

from ..attention_coupling.unet_attention_phase_session import (
    StandardUnetAttentionPhaseSession,
)
from ..attention_coupling.unet_attention_state import StandardUnetAttentionState
from ..attention_coupling.unet_attn2_execution_resolver import (
    StandardUnetAttn2ExecutionResolver,
)
from ..model_patcher_mutations import (
    ModelKeyedCallbackMutation,
    ModelKeyedWrapperMutation,
)
from ..ppm_negpip_interop import PpmNegpipInterop
from .standard_unet_cold_sampling import (
    StandardUnetColdSamplingDiagnosticsMutation,
)
from .standard_unet_native_admission import StandardUnetNativeLoraAdmission
from .standard_unet_variant_base_attention import StandardUnetVariantBaseAttention
from .standard_unet_variant_conditioning import (
    StandardUnetVariantConditioningResolver,
)
from .standard_unet_variant_execution import StandardUnetVariantExecution
from .standard_unet_variant_execution_session import (
    StandardUnetVariantExecutionActivation,
)
from .standard_unet_variant_template import StandardUnetVariantTemplate

_DETACH_KEY = "simple_syrup.standard_unet_regional_variants"
_EXECUTION_WRAPPER_KEY = "simple_syrup.standard_unet_variant_execution"


@dataclass(frozen=True, slots=True)
class StandardUnetVariantRuntimeMutation:
    """Construct and install one exact model-active standard-UNet runtime."""

    state: StandardUnetAttentionState
    admission: StandardUnetNativeLoraAdmission
    attention_phase: StandardUnetAttentionPhaseSession
    template: StandardUnetVariantTemplate
    negpip: PpmNegpipInterop | None = None

    def apply(self, model: object) -> None:
        """Build persistent variants before installing the private root clone."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Standard UNet variant runtime requires a MODEL.")
        if self.admission.resolution is None:
            raise ValueError("Standard UNet variant runtime requires resolution.")
        if not isinstance(self.template, StandardUnetVariantTemplate):
            raise TypeError("Standard UNet variant runtime requires a template.")
        execution = StandardUnetVariantExecution(
            base_diffusion=self.template.root.module,
            plan=self.admission.adaptation.plan,
            topology=self.template.topology,
            mask_bank=self.state.plan.mask_bank,
            variant_resolver=self.template,
            conditioning=StandardUnetVariantConditioningResolver(
                self.state.execution_context
            ),
            attention_phase=self.attention_phase,
            base_attention=StandardUnetVariantBaseAttention(
                StandardUnetAttn2ExecutionResolver(self.state),
                negpip=self.negpip,
            ),
        )
        execution.prime()
        StandardUnetColdSamplingDiagnosticsMutation().apply(model)
        ModelKeyedWrapperMutation(
            WrappersMP.DIFFUSION_MODEL,
            _EXECUTION_WRAPPER_KEY,
            StandardUnetVariantExecutionActivation(
                self.template.execution_session,
                execution,
            ),
        ).apply(model)
        ModelKeyedCallbackMutation(
            CallbacksMP.ON_DETACH,
            _DETACH_KEY,
            execution.clear,
        ).apply(model)
