# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable regional feature requests, capabilities, and admissions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .regional_model_capabilities import RegionalModelCapabilities


class RegionalFeature(StrEnum):
    """Identify one independently admitted regional sampling feature."""

    FULL_CONTEXT_MASKED_CONDITIONING = "full_context_masked_conditioning"
    ATTENTION_COUPLING = "attention_coupling"
    SPATIAL_MODEL_PATCH = "spatial_model_patch"
    CONTROL = "control"
    GLIGEN = "gligen"
    REFERENCE_LATENTS = "reference_latents"


@dataclass(frozen=True, slots=True)
class RegionalFeatureRequest:
    """Describe the complete immutable regional feature intent for one sample."""

    features: frozenset[RegionalFeature] = frozenset()

    def __post_init__(self) -> None:
        """Require an immutable set containing only typed regional features."""

        _validate_features(self.features, value_name="Regional feature request")

    def with_feature(self, feature: RegionalFeature) -> RegionalFeatureRequest:
        """Return a new request containing one additional typed feature."""

        if not isinstance(feature, RegionalFeature):
            raise TypeError("Requested regional feature must be a RegionalFeature.")
        return RegionalFeatureRequest(self.features | {feature})


@dataclass(frozen=True, slots=True)
class RegionalSamplerCapabilities:
    """Describe the complete regional feature set implemented by one sampler."""

    features: frozenset[RegionalFeature]

    def __post_init__(self) -> None:
        """Require an immutable set containing only typed regional features."""

        _validate_features(self.features, value_name="Regional sampler capabilities")


@dataclass(frozen=True, slots=True)
class RegionalCapabilityAdmission:
    """Record one complete successful request admission for downstream use."""

    request: RegionalFeatureRequest
    admitted_features: frozenset[RegionalFeature]
    model_capabilities: RegionalModelCapabilities | None

    def __post_init__(self) -> None:
        """Reject partial, untyped, or model-incomplete admission state."""

        if not isinstance(self.request, RegionalFeatureRequest):
            raise TypeError(
                "Regional admission request must be RegionalFeatureRequest."
            )
        _validate_features(
            self.admitted_features,
            value_name="Admitted regional features",
        )
        if self.admitted_features != self.request.features:
            raise ValueError("Regional capability admission cannot be partial.")
        if self.model_capabilities is not None and not isinstance(
            self.model_capabilities,
            RegionalModelCapabilities,
        ):
            raise TypeError(
                "Regional admission model capabilities must be "
                "RegionalModelCapabilities."
            )
        if self.admitted_features & MODEL_DEPENDENT_REGIONAL_FEATURES:
            if self.model_capabilities is None:
                raise ValueError(
                    "Model-dependent regional features require model capabilities."
                )

    def supports(self, feature: RegionalFeature) -> bool:
        """Return whether one typed feature was admitted for this sample."""

        if not isinstance(feature, RegionalFeature):
            raise TypeError("Regional feature query must be a RegionalFeature.")
        return feature in self.admitted_features


MODEL_DEPENDENT_REGIONAL_FEATURES = frozenset(
    {
        RegionalFeature.ATTENTION_COUPLING,
        RegionalFeature.SPATIAL_MODEL_PATCH,
        RegionalFeature.CONTROL,
        RegionalFeature.GLIGEN,
        RegionalFeature.REFERENCE_LATENTS,
    }
)


def _validate_features(
    features: frozenset[RegionalFeature],
    *,
    value_name: str,
) -> None:
    """Validate one immutable typed regional feature set."""

    if not isinstance(features, frozenset):
        raise TypeError(f"{value_name} must use an immutable frozenset.")
    if not all(isinstance(feature, RegionalFeature) for feature in features):
        raise TypeError(f"{value_name} must contain RegionalFeature values.")


EMPTY_REGIONAL_FEATURE_REQUEST = RegionalFeatureRequest()
EMPTY_REGIONAL_CAPABILITY_ADMISSION = RegionalCapabilityAdmission(
    request=EMPTY_REGIONAL_FEATURE_REQUEST,
    admitted_features=frozenset(),
    model_capabilities=None,
)
CONTEXTUAL_DIFFUSION_REGIONAL_SAMPLER_CAPABILITIES = RegionalSamplerCapabilities(
    frozenset(
        {
            RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING,
            RegionalFeature.ATTENTION_COUPLING,
        }
    )
)
TILED_DIFFUSION_REGIONAL_SAMPLER_CAPABILITIES = RegionalSamplerCapabilities(
    frozenset(
        {
            RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING,
            RegionalFeature.ATTENTION_COUPLING,
        }
    )
)
