# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Admit complete standard-UNet regional LoRA surfaces before model loading."""

from __future__ import annotations

from dataclasses import dataclass

import comfy.model_patcher
from torch import nn

from ..attention_coupling.family_admission import (
    AttentionCouplingFamilyAdmission,
)
from ..regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from .comfy_adapter_resolver import COMFY_REGIONAL_ADAPTER_RESOLVER
from .execution_cache import RegionalLoraExecutionCache
from .resolved_operation_translator import COMFY_RESOLVED_OPERATION_TRANSLATOR
from .standard_unet_target_capabilities import (
    STANDARD_UNET_TARGET_CAPABILITY_CLASSIFIER,
)
from .target_binder import REGIONAL_LORA_TARGET_BINDER
from .target_binding import (
    BoundRegionalLoraSpatialCapability,
    RegionalLoraBindingResult,
)


@dataclass(frozen=True, slots=True)
class StandardUnetOperationAdmission(AttentionCouplingFamilyAdmission):
    """Retain complete target binding and runtime consumer-role evidence."""

    binding: RegionalLoraBindingResult | None
    module_roles: dict[str, BoundRegionalLoraSpatialCapability]
    cache: RegionalLoraExecutionCache | None

    def __post_init__(self) -> None:
        """Require either one empty admission or a complete executable surface."""

        AttentionCouplingFamilyAdmission.__post_init__(self)
        if not self.adaptation.plan.adapters:
            if self.binding is not None or self.module_roles or self.cache is not None:
                raise ValueError("Empty standard admission cannot retain operations.")
            return
        if (
            not isinstance(self.binding, RegionalLoraBindingResult)
            or not self.binding.admissible
            or not self.binding.entries
        ):
            raise ValueError("Standard admission requires complete target binding.")
        if not self.module_roles or self.cache is None:
            raise ValueError("Standard admission requires roles and execution cache.")


class StandardUnetOperationPreparation:
    """Compose existing resolution, translation, binding, and role authorities."""

    def admit(
        self,
        model: object,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> StandardUnetOperationAdmission:
        """Return complete immutable evidence or fail before model load/mutation."""

        if not isinstance(adaptation, RegionalLoraPlanAdaptation):
            raise TypeError("Standard UNet operation admission requires adaptation.")
        if not adaptation.plan.adapters:
            return StandardUnetOperationAdmission(adaptation, None, {}, None)
        if not isinstance(model, comfy.model_patcher.ModelPatcher):
            raise TypeError("Standard UNet operation admission requires a MODEL.")
        graph_root = model.model
        if not isinstance(graph_root, nn.Module):
            raise TypeError("Standard UNet MODEL graph must be an nn.Module.")
        capabilities = STANDARD_UNET_TARGET_CAPABILITY_CLASSIFIER.classify(graph_root)
        resolution = COMFY_REGIONAL_ADAPTER_RESOLVER.resolve(
            adaptation,
            model=model,
        )
        operations = COMFY_RESOLVED_OPERATION_TRANSLATOR.translate(resolution)
        binding = REGIONAL_LORA_TARGET_BINDER.bind(
            source=model,
            candidate=model,
            resolution=resolution,
            operations=operations,
            linear_spatial_capabilities=capabilities.linear_roles,
        )
        if not binding.admissible:
            messages = tuple(issue.message for issue in binding.issues)
            raise ValueError(
                f"Standard UNet regional LoRA target admission failed: {messages!r}."
            )
        unavailable = tuple(
            entry.descriptor.target.parameter_path
            for entry in binding.entries
            if entry.spatial_capability
            in (
                BoundRegionalLoraSpatialCapability.GLOBAL_ONLY,
                BoundRegionalLoraSpatialCapability.UNSUPPORTED,
            )
        )
        if unavailable:
            raise ValueError(
                "Standard UNet regional LoRA targets lack executable consumer roles: "
                f"{unavailable!r}."
            )
        module_roles: dict[str, BoundRegionalLoraSpatialCapability] = {}
        for entry in binding.entries:
            path = entry.descriptor.target.model_target
            role = entry.spatial_capability
            previous = module_roles.setdefault(path, role)
            if previous is not role:
                raise ValueError(
                    f"Standard UNet operation {path!r} has conflicting roles."
                )
        return StandardUnetOperationAdmission(
            adaptation,
            binding,
            module_roles,
            RegionalLoraExecutionCache(),
        )


STANDARD_UNET_OPERATION_PREPARATION = StandardUnetOperationPreparation()
