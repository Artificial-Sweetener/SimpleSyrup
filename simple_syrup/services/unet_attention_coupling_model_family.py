# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt shared Attention Coupling preparation to the standard-UNet backend."""

from __future__ import annotations

from typing import ClassVar

import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
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
from ..runtime.regional_lora.standard_unet_operation_preparation import (
    StandardUnetOperationAdmission,
    StandardUnetOperationPreparation,
)
from ..runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation

_STANDARD_UNET_BACKEND_IDENTITY = f"{UNetModel.__module__}.{UNetModel.__qualname__}"


class StandardUnetAttentionCouplingModelFamily:
    """Own standard-UNet latent, adapter, state, and derivation policy."""

    backend_class: ClassVar[type[StandardUnetAttentionBackend]] = (
        StandardUnetAttentionBackend
    )
    operation_preparation_class: ClassVar[type[StandardUnetOperationPreparation]] = (
        StandardUnetOperationPreparation
    )

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
        """Resolve and bind the complete regional operation surface before load."""

        if not isinstance(adaptation, RegionalLoraPlanAdaptation):
            raise TypeError(
                "Standard UNet Attention Coupling requires regional adaptation."
            )
        return self.operation_preparation_class().admit(model, adaptation)

    def derive(
        self,
        *,
        model: object,
        processed_plan: ProcessedRegionalAttentionPlan,
        admission: AttentionCouplingFamilyAdmission,
        region_strengths: tuple[float, ...],
        latent_batch_size: int,
    ) -> object:
        """Build shared diagnostics state and derive the paired attn2 backend."""

        if not isinstance(admission, StandardUnetOperationAdmission):
            raise TypeError("Standard UNet derivation requires family admission.")
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
            )
            .model
        )
