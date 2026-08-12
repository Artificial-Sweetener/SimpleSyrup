# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Admit regional feature requests against sampler and model capabilities."""

from __future__ import annotations

from typing import ClassVar, Protocol

from ..domain.regional_features import (
    MODEL_DEPENDENT_REGIONAL_FEATURES,
    RegionalCapabilityAdmission,
    RegionalFeature,
    RegionalFeatureRequest,
    RegionalSamplerCapabilities,
)
from ..domain.regional_model_capabilities import (
    RegionalControlGligenPolicy,
    RegionalModelCapabilities,
    RegionalReferenceLatentPolicy,
    RegionalSpatialPatchSupport,
)
from ..runtime.regional_model_capabilities import RegionalModelCapabilityRegistry


class RegionalModelCapabilityProvider(Protocol):
    """Resolve one model into immutable regional capabilities."""

    def capabilities_for(self, model: object) -> RegionalModelCapabilities:
        """Return capabilities for one supported Comfy MODEL."""


class RegionalCapabilityAdmissionService:
    """Own complete feature admission across sampler and model boundaries."""

    model_registry_class: ClassVar[type[RegionalModelCapabilityProvider]] = (
        RegionalModelCapabilityRegistry
    )

    def admit(
        self,
        *,
        request: RegionalFeatureRequest,
        sampler_capabilities: RegionalSamplerCapabilities,
        model: object,
    ) -> RegionalCapabilityAdmission:
        """Return a complete admission or report every unsupported feature."""

        if not isinstance(request, RegionalFeatureRequest):
            raise TypeError("Regional feature request must be RegionalFeatureRequest.")
        if not isinstance(sampler_capabilities, RegionalSamplerCapabilities):
            raise TypeError(
                "Regional sampler capabilities must be RegionalSamplerCapabilities."
            )
        unsupported = request.features - sampler_capabilities.features
        if unsupported:
            raise ValueError(
                "Regional sampler does not support requested features: "
                f"{_feature_names(unsupported)}."
            )

        model_features = request.features & MODEL_DEPENDENT_REGIONAL_FEATURES
        model_capabilities = None
        rejected_by_model: set[RegionalFeature] = set()
        if model_features:
            model_capabilities = self.model_registry_class().capabilities_for(model)
            if (
                RegionalFeature.SPATIAL_MODEL_PATCH in model_features
                and model_capabilities.spatial_patch_support
                is not RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS
            ):
                rejected_by_model.add(RegionalFeature.SPATIAL_MODEL_PATCH)
            if (
                model_capabilities.control_gligen_policy
                is RegionalControlGligenPolicy.REJECT
            ):
                rejected_by_model.update(
                    model_features & {RegionalFeature.CONTROL, RegionalFeature.GLIGEN}
                )
            if (
                RegionalFeature.REFERENCE_LATENTS in model_features
                and model_capabilities.reference_latent_policy
                is RegionalReferenceLatentPolicy.REJECT
            ):
                rejected_by_model.add(RegionalFeature.REFERENCE_LATENTS)
        if rejected_by_model:
            raise ValueError(
                "Regional model does not support requested features: "
                f"{_feature_names(rejected_by_model)}."
            )
        return RegionalCapabilityAdmission(
            request=request,
            admitted_features=request.features,
            model_capabilities=model_capabilities,
        )


def _feature_names(features: set[RegionalFeature] | frozenset[RegionalFeature]) -> str:
    """Return stable feature names for actionable admission diagnostics."""

    return ", ".join(sorted(feature.value for feature in features))
