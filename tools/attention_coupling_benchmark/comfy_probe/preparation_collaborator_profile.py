# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Profile exact model-preparation collaborators without copying orchestration."""

from __future__ import annotations

from typing import ClassVar

import torch

from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.raw_regional_attention import RawRegionalAttentionPlan
from simple_syrup.runtime.attention_coupling.context_validation import (
    RegionalContextValidator,
)
from simple_syrup.runtime.comfy_conditioning_model_loader import (
    ComfyConditioningModelLoader,
)
from simple_syrup.runtime.comfy_conditioning_processing import (
    ComfyRegionalConditioningProcessor,
)
from simple_syrup.runtime.regional_lora_conditioning_adapter import (
    RegionalLoraConditioningAdapter,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropValidator,
)
from simple_syrup.services.attention_coupling_model_family_selector import (
    AttentionCouplingModelFamilySelector,
)
from simple_syrup.services.attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from simple_syrup.services.attention_coupling_preparation_service import (
    AttentionCouplingPreparation,
    AttentionCouplingPreparationService,
)

from .interop_validation_profile import ProfiledRegionalModelPatchInteropValidator
from .model_family_profile import ProfiledAttentionCouplingModelFamilySelector
from .synchronized_phase_timing import measure_synchronized_phase, model_device


class ProfiledComfyConditioningModelLoader(ComfyConditioningModelLoader):
    """Time exact source-model loading through the production owner."""

    def load(self, model: object) -> None:
        """Delegate source loading while recording its synchronized duration."""

        with measure_synchronized_phase(
            "source_model_load",
            device=model_device(model),
        ):
            return super().load(model)


class ProfiledRegionalLoraConditioningAdapter(RegionalLoraConditioningAdapter):
    """Time exact regional-LoRA collection and adaptation."""

    def adapt(
        self,
        plan: RawRegionalAttentionPlan,
        *,
        model: object,
    ) -> RegionalLoraPlanAdaptation:
        """Delegate adaptation while recording its CPU-owned duration."""

        with measure_synchronized_phase(
            "regional_lora_adaptation",
            device=model_device(model),
        ):
            return super().adapt(plan, model=model)


class ProfiledAttentionCouplingPreparationService(AttentionCouplingPreparationService):
    """Time exact architecture-neutral regional-plan preparation."""

    def prepare(self, plan: RawRegionalAttentionPlan) -> AttentionCouplingPreparation:
        """Delegate plan preparation while recording its CPU duration."""

        with measure_synchronized_phase(
            "regional_plan_preparation",
            device=None,
        ):
            return super().prepare(plan)


class ProfiledComfyRegionalConditioningProcessor(ComfyRegionalConditioningProcessor):
    """Time exact Comfy conditioning conversion and encoding."""

    def process(
        self,
        preparation: AttentionCouplingPreparation,
        *,
        model: object,
        noise: torch.Tensor,
        device: torch.device,
        context_validator: RegionalContextValidator,
    ) -> ProcessedRegionalAttentionPlan:
        """Delegate conditioning processing with synchronized device timing."""

        with measure_synchronized_phase(
            "conditioning_processing",
            device=device,
        ):
            return super().process(
                preparation,
                model=model,
                noise=noise,
                device=device,
                context_validator=context_validator,
            )


class ProfiledAttentionCouplingModelPreparationService(
    AttentionCouplingModelPreparationService
):
    """Run production orchestration with exact-delegating timed collaborators."""

    model_loader_class: ClassVar[type[ComfyConditioningModelLoader]] = (
        ProfiledComfyConditioningModelLoader
    )
    lora_adapter_class: ClassVar[type[RegionalLoraConditioningAdapter]] = (
        ProfiledRegionalLoraConditioningAdapter
    )
    preparation_service_class: ClassVar[type[AttentionCouplingPreparationService]] = (
        ProfiledAttentionCouplingPreparationService
    )
    conditioning_processor_class: ClassVar[type[ComfyRegionalConditioningProcessor]] = (
        ProfiledComfyRegionalConditioningProcessor
    )
    interop_validator_class: ClassVar[type[RegionalModelPatchInteropValidator]] = (
        ProfiledRegionalModelPatchInteropValidator
    )
    model_family_selector_class: ClassVar[
        type[AttentionCouplingModelFamilySelector]
    ] = ProfiledAttentionCouplingModelFamilySelector
