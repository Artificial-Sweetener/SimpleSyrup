# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Admit model-active standard-UNet LoRAs for native Comfy patching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..attention_coupling.family_admission import AttentionCouplingFamilyAdmission
from ..regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from .comfy_adapter_resolution import (
    ComfyAdapterResolutionIssueCode,
    ComfyRegionalLoraResolution,
)
from .comfy_adapter_resolver import COMFY_REGIONAL_ADAPTER_RESOLVER
from .standard_unet_cold_diagnostics import (
    STANDARD_UNET_COLD_PATH_DIAGNOSTICS,
    StandardUnetColdStage,
)

_BLOCKING_ISSUE_CODES = frozenset(
    {
        ComfyAdapterResolutionIssueCode.RESOLUTION_FAILED,
        ComfyAdapterResolutionIssueCode.INVALID_TARGET_PATH,
    }
)


class NativeAdapterResolver(Protocol):
    """Expose installed-Comfy model target resolution."""

    def resolve(
        self,
        adaptation: RegionalLoraPlanAdaptation,
        *,
        model: object,
    ) -> ComfyRegionalLoraResolution:
        """Return complete normalized adapter evidence."""

        ...


@dataclass(frozen=True, slots=True)
class StandardUnetNativeLoraAdmission(AttentionCouplingFamilyAdmission):
    """Retain complete native model-target evidence for standard UNets."""

    resolution: ComfyRegionalLoraResolution | None

    def __post_init__(self) -> None:
        """Require empty or complete evidence according to adapter presence."""

        AttentionCouplingFamilyAdmission.__post_init__(self)
        if self.adaptation.plan.adapters:
            if not isinstance(self.resolution, ComfyRegionalLoraResolution):
                raise ValueError("Native standard-UNet LoRAs require resolution.")
        elif self.resolution is not None:
            raise ValueError("Prompt-only native admission cannot retain resolution.")


class StandardUnetNativeLoraAdmissionService:
    """Validate native Comfy target coverage without operation translation."""

    def __init__(
        self,
        resolver: NativeAdapterResolver = COMFY_REGIONAL_ADAPTER_RESOLVER,
    ) -> None:
        """Retain the installed-Comfy resolution authority."""

        self._resolver = resolver

    def admit(
        self,
        model: object,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> StandardUnetNativeLoraAdmission:
        """Return complete evidence or fail before loading and mutation."""

        if not isinstance(adaptation, RegionalLoraPlanAdaptation):
            raise TypeError("Native standard-UNet admission requires adaptation.")
        if not adaptation.plan.adapters:
            return StandardUnetNativeLoraAdmission(adaptation, None)
        device = getattr(model, "load_device", None)
        measured_device = device if isinstance(device, torch.device) else None
        with STANDARD_UNET_COLD_PATH_DIAGNOSTICS.measure(
            StandardUnetColdStage.ADMISSION_RESOLUTION,
            device=measured_device,
        ) as metadata:
            resolution = self._resolver.resolve(adaptation, model=model)
            self._validate(adaptation, resolution)
            metadata["adapter_count"] = len(resolution.adapters)
            metadata["target_count"] = sum(
                len(result.model_targets) for result in resolution.adapters
            )
        return StandardUnetNativeLoraAdmission(adaptation, resolution)

    @staticmethod
    def _validate(
        adaptation: RegionalLoraPlanAdaptation,
        resolution: ComfyRegionalLoraResolution,
    ) -> None:
        """Require one usable native model target surface per adapter use."""

        if not isinstance(resolution, ComfyRegionalLoraResolution):
            raise TypeError("Native adapter resolver returned invalid evidence.")
        if len(resolution.adapters) != len(adaptation.plan.adapters):
            raise ValueError(
                "Standard UNet native adapter admission failed: resolver result "
                "count does not match the adapted use count."
            )
        blocking = tuple(
            issue for issue in resolution.issues if issue.code in _BLOCKING_ISSUE_CODES
        )
        if blocking:
            messages = tuple(issue.message for issue in blocking)
            raise ValueError(
                f"Standard UNet native adapter admission failed: {messages!r}."
            )
        for expected, result in zip(
            adaptation.plan.adapters,
            resolution.adapters,
            strict=True,
        ):
            if result.adapter is not expected and result.adapter != expected:
                raise ValueError(
                    "Standard UNet native adapter admission failed: resolver "
                    "changed adapter ownership or order."
                )
            if not result.model_targets:
                raise ValueError(
                    "Standard UNet regional adapter "
                    f"{expected.composition_index} resolved no model targets."
                )


STANDARD_UNET_NATIVE_LORA_ADMISSION_SERVICE = StandardUnetNativeLoraAdmissionService()
