# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare one reusable admitted Attention Coupling model and base pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ..domain.raw_regional_attention import (
    RawRegionalAttentionPlan,
    build_raw_regional_attention_plan,
)
from ..domain.regional_attention_execution import RegionalAttentionExecutionMode
from ..domain.regional_features import (
    RegionalFeature,
    RegionalFeatureRequest,
    RegionalSamplerCapabilities,
)
from ..domain.regional_mask_bank import RegionalMaskBank
from ..masking.regional_prompt_masks import build_regional_mask_bank
from ..runtime.comfy_conditioning_model_loader import ComfyConditioningModelLoader
from ..runtime.comfy_conditioning_processing import (
    ComfyRegionalConditioningProcessor,
)
from ..runtime.regional_lora_conditioning_adapter import (
    RegionalLoraConditioningAdapter,
)
from ..runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropValidator,
)
from .attention_coupling_model_family_selector import (
    AttentionCouplingModelFamilySelector,
)
from .attention_coupling_preparation_service import (
    AttentionCouplingPreparationService,
)
from .regional_capability_admission_service import (
    RegionalCapabilityAdmissionService,
)

_ATTENTION_FEATURES = frozenset({RegionalFeature.ATTENTION_COUPLING})
_ATTENTION_REQUEST = RegionalFeatureRequest(_ATTENTION_FEATURES)
_ATTENTION_CAPABILITIES = RegionalSamplerCapabilities(_ATTENTION_FEATURES)


@dataclass(frozen=True, slots=True)
class PreparedAttentionCouplingModel:
    """Retain a derived model, base conditioning pair, and canonical mask bank."""

    model: Any
    positive: object
    negative: object
    mask_bank: RegionalMaskBank


class AttentionCouplingModelPreparationService:
    """Own the model-neutral prepare-once Attention Coupling sequence."""

    capability_service_class: ClassVar[type[RegionalCapabilityAdmissionService]] = (
        RegionalCapabilityAdmissionService
    )
    preparation_service_class: ClassVar[type[AttentionCouplingPreparationService]] = (
        AttentionCouplingPreparationService
    )
    model_loader_class: ClassVar[type[ComfyConditioningModelLoader]] = (
        ComfyConditioningModelLoader
    )
    conditioning_processor_class: ClassVar[type[ComfyRegionalConditioningProcessor]] = (
        ComfyRegionalConditioningProcessor
    )
    lora_adapter_class: ClassVar[type[RegionalLoraConditioningAdapter]] = (
        RegionalLoraConditioningAdapter
    )
    interop_validator_class: ClassVar[type[RegionalModelPatchInteropValidator]] = (
        RegionalModelPatchInteropValidator
    )
    model_family_selector_class: ClassVar[
        type[AttentionCouplingModelFamilySelector]
    ] = AttentionCouplingModelFamilySelector

    def prepare(
        self,
        *,
        model: Any,
        positive: object,
        negative: object,
        region_masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
        latent_image: dict[str, Any],
        execution_mode: RegionalAttentionExecutionMode,
    ) -> PreparedAttentionCouplingModel:
        """Return one admitted derived model ready for spatial sampling."""

        samples = self._latent_samples(latent_image)
        admission = self.capability_service_class().admit(
            request=_ATTENTION_REQUEST,
            sampler_capabilities=_ATTENTION_CAPABILITIES,
            model=model,
        )
        capabilities = admission.model_capabilities
        if capabilities is None:
            raise ValueError(
                "Attention Coupling admission did not return model capabilities."
            )
        interop_validator = self.interop_validator_class()
        interop_report = interop_validator.validate(model, capabilities)
        model_family = self.model_family_selector_class().select(capabilities)
        model_family.validate_latent(samples)
        mask_bank = build_regional_mask_bank(
            region_masks,
            feather=region_mask_feather,
            canvas_height=int(samples.shape[-2]),
            canvas_width=int(samples.shape[-1]),
        )
        raw = build_raw_regional_attention_plan(
            positive=positive,
            negative=negative,
            mask_bank=mask_bank,
        )
        base_model = getattr(model, "model", None)
        adaptation = self.lora_adapter_class().adapt(raw, model=base_model)
        model_family.validate_adaptation(adaptation)
        raw_with_loras = RawRegionalAttentionPlan(
            raw.positive,
            raw.negative,
            raw.mask_bank,
            adaptation.plan,
        )
        preparation = self.preparation_service_class().prepare(raw_with_loras)
        device = torch.device(getattr(model, "load_device", "cpu"))
        self.model_loader_class().load(model)
        processed = self.conditioning_processor_class().process(
            preparation,
            model=model,
            noise=samples.to(device),
            device=device,
            context_validator=model_family.context_validator,
        )
        interop_validator.validate_execution(
            interop_report,
            processed,
            execution_mode,
        )
        region_strengths = (float(regional_prompt_weight),) * mask_bank.region_count
        derived_model = model_family.derive(
            model=model,
            processed_plan=processed,
            adaptation=adaptation,
            region_strengths=region_strengths,
            latent_batch_size=int(samples.shape[0]),
        )
        return PreparedAttentionCouplingModel(
            derived_model,
            preparation.positive,
            preparation.negative,
            mask_bank,
        )

    @staticmethod
    def _latent_samples(latent_image: object) -> torch.Tensor:
        """Return validated floating latent samples before any model work."""

        if not isinstance(latent_image, dict):
            raise TypeError("Attention Coupling latent_image must be a dictionary.")
        samples = latent_image.get("samples")
        if not isinstance(samples, torch.Tensor) or not samples.is_floating_point():
            raise TypeError(
                "Attention Coupling latent_image must contain floating samples."
            )
        if samples.ndim not in (4, 5):
            raise ValueError("Attention Coupling latent samples must be 4D or 5D.")
        return samples
