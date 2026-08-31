# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare one reusable admitted Attention Coupling model and base pair."""

from __future__ import annotations

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
from ..masking.regional_prompt_masks import build_regional_mask_bank
from ..runtime.comfy_conditioning_model_loader import ComfyConditioningModelLoader
from ..runtime.comfy_conditioning_processing import (
    ComfyRegionalConditioningProcessor,
)
from ..runtime.comfy_latent_normalization import ComfyLatentNormalizer
from ..runtime.regional_lora_conditioning_adapter import (
    RegionalLoraConditioningAdapter,
)
from ..runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropReport,
    RegionalModelPatchInteropValidator,
)
from .attention_coupling_model_family import (
    AttentionCouplingModelFamily,
    AttentionCouplingPreparedModelReuse,
)
from .attention_coupling_model_family_selector import (
    AttentionCouplingModelFamilySelector,
)
from .attention_coupling_preparation_service import (
    AttentionCouplingPreparationService,
)
from .attention_coupling_prepared_model_cache import (
    ATTENTION_COUPLING_PREPARED_MODEL_CACHE,
    AttentionCouplingPreparedModelCache,
    AttentionCouplingPreparedRequest,
)
from .prepared_attention_coupling_model import PreparedAttentionCouplingModel
from .regional_capability_admission_service import (
    RegionalCapabilityAdmissionService,
)

_ATTENTION_FEATURES = frozenset({RegionalFeature.ATTENTION_COUPLING})
_ATTENTION_REQUEST = RegionalFeatureRequest(_ATTENTION_FEATURES)
_ATTENTION_CAPABILITIES = RegionalSamplerCapabilities(_ATTENTION_FEATURES)


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
    latent_normalizer_class: ClassVar[type[ComfyLatentNormalizer]] = (
        ComfyLatentNormalizer
    )
    model_family_selector_class: ClassVar[
        type[AttentionCouplingModelFamilySelector]
    ] = AttentionCouplingModelFamilySelector
    prepared_model_cache: ClassVar[AttentionCouplingPreparedModelCache] = (
        ATTENTION_COUPLING_PREPARED_MODEL_CACHE
    )

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
        samples = self.latent_normalizer_class().normalize(
            model=model,
            samples=samples,
            spatial_downscale_ratio=latent_image.get(
                "downscale_ratio_spacial",
                None,
            ),
            temporal_downscale_ratio=latent_image.get(
                "downscale_ratio_temporal",
                None,
            ),
        )
        model_family.validate_latent(samples)

        def prepare_uncached() -> PreparedAttentionCouplingModel:
            """Execute this validated request through the existing miss path."""

            return self._prepare_uncached(
                model=model,
                positive=positive,
                negative=negative,
                region_masks=region_masks,
                regional_prompt_weight=regional_prompt_weight,
                region_mask_feather=region_mask_feather,
                samples=samples,
                execution_mode=execution_mode,
                interop_validator=interop_validator,
                interop_report=interop_report,
                model_family=model_family,
            )

        policy = model_family.prepared_model_reuse
        if policy is AttentionCouplingPreparedModelReuse.DISABLED:
            return prepare_uncached()
        request = AttentionCouplingPreparedRequest.capture(
            model=model,
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
            latent_image=latent_image,
            execution_mode=execution_mode,
        )
        return self.prepared_model_cache.resolve(
            request=request,
            policy=policy,
            prepare=prepare_uncached,
        )

    def _prepare_uncached(
        self,
        *,
        model: object,
        positive: object,
        negative: object,
        region_masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
        samples: torch.Tensor,
        execution_mode: RegionalAttentionExecutionMode,
        interop_validator: RegionalModelPatchInteropValidator,
        interop_report: RegionalModelPatchInteropReport,
        model_family: AttentionCouplingModelFamily,
    ) -> PreparedAttentionCouplingModel:
        """Execute the existing complete preparation sequence for one cache miss."""

        mask_bank = build_regional_mask_bank(
            region_masks,
            feather=region_mask_feather,
            canvas_height=int(samples.shape[-2]),
            canvas_width=int(samples.shape[-1]),
        )
        region_strengths = (float(regional_prompt_weight),) * mask_bank.region_count
        raw = build_raw_regional_attention_plan(
            positive=positive,
            negative=negative,
            mask_bank=mask_bank,
        )
        self.model_loader_class().load(model)
        base_model = getattr(model, "model", None)
        adaptation = self.lora_adapter_class().adapt(raw, model=base_model)
        family_admission = model_family.admit_adaptation(model, adaptation)
        raw_with_loras = RawRegionalAttentionPlan(
            raw.positive,
            raw.negative,
            raw.mask_bank,
            adaptation.plan,
        )
        sampler_conditioning = model_family.prepare_sampler_conditioning(
            raw_with_loras,
            region_strengths,
        )
        preparation = self.preparation_service_class().prepare(raw_with_loras)
        device = torch.device(getattr(model, "load_device", "cpu"))
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
        derived_model = model_family.derive(
            model=model,
            processed_plan=processed,
            admission=family_admission,
            interop_report=interop_report,
            region_strengths=region_strengths,
            latent_batch_size=int(samples.shape[0]),
        )
        return PreparedAttentionCouplingModel(
            derived_model,
            sampler_conditioning.positive,
            sampler_conditioning.negative,
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
