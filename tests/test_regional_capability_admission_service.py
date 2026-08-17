# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify regional requests are admitted against sampler and model policy."""

from __future__ import annotations

from typing import ClassVar

import pytest

from simple_syrup.domain.regional_features import (
    RegionalFeature,
    RegionalFeatureRequest,
    RegionalSamplerCapabilities,
)
from simple_syrup.domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalAttentionTopology,
    RegionalControlGligenPolicy,
    RegionalLatentLayout,
    RegionalModelCapabilities,
    RegionalModelFamily,
    RegionalPatchConflict,
    RegionalReferenceLatentPolicy,
    RegionalSpatialPatchSupport,
)
from simple_syrup.services.regional_capability_admission_service import (
    RegionalCapabilityAdmissionService,
)


class _Registry:
    """Return fixed Anima capabilities while recording detector use."""

    calls: ClassVar[list[object]] = []

    def capabilities_for(self, model: object) -> RegionalModelCapabilities:
        """Record the model and return one fixed admitted family."""

        self.calls.append(model)
        return _anima_capabilities()


class _AdmissionService(RegionalCapabilityAdmissionService):
    """Use the deterministic registry at the external model boundary."""

    model_registry_class = _Registry


def test_sampler_only_request_skips_model_detection() -> None:
    """Admit full-context masks without restricting ordinary model families."""

    _Registry.calls.clear()
    request = _request(RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING)

    admission = _AdmissionService().admit(
        request=request,
        sampler_capabilities=RegionalSamplerCapabilities(request.features),
        model=object(),
    )

    assert admission.admitted_features == request.features
    assert admission.model_capabilities is None
    assert _Registry.calls == []


def test_attention_and_spatial_patch_request_uses_model_registry_once() -> None:
    """Attach defensive model capabilities to admitted model-dependent features."""

    _Registry.calls.clear()
    model = object()
    request = RegionalFeatureRequest(
        frozenset(
            {RegionalFeature.ATTENTION_COUPLING, RegionalFeature.SPATIAL_MODEL_PATCH}
        )
    )

    admission = _AdmissionService().admit(
        request=request,
        sampler_capabilities=RegionalSamplerCapabilities(request.features),
        model=model,
    )

    assert admission.model_capabilities == _anima_capabilities()
    assert _Registry.calls == [model]


def test_sampler_rejection_reports_every_unsupported_feature() -> None:
    """Reject the full unsupported set before attempting model detection."""

    _Registry.calls.clear()
    request = RegionalFeatureRequest(
        frozenset({RegionalFeature.CONTROL, RegionalFeature.GLIGEN})
    )

    with pytest.raises(ValueError) as captured:
        _AdmissionService().admit(
            request=request,
            sampler_capabilities=RegionalSamplerCapabilities(frozenset()),
            model=object(),
        )

    assert "control, gligen" in str(captured.value)
    assert _Registry.calls == []


def test_model_rejection_reports_control_gligen_and_reference_latents() -> None:
    """Apply every explicit P3.6 conditioning policy in one admission result."""

    request = RegionalFeatureRequest(
        frozenset(
            {
                RegionalFeature.ATTENTION_COUPLING,
                RegionalFeature.CONTROL,
                RegionalFeature.GLIGEN,
                RegionalFeature.REFERENCE_LATENTS,
            }
        )
    )

    with pytest.raises(ValueError) as captured:
        _AdmissionService().admit(
            request=request,
            sampler_capabilities=RegionalSamplerCapabilities(request.features),
            model=object(),
        )

    message = str(captured.value)
    assert "control" in message
    assert "gligen" in message
    assert "reference_latents" in message


def _request(feature: RegionalFeature) -> RegionalFeatureRequest:
    """Return one single-feature request."""

    return RegionalFeatureRequest(frozenset({feature}))


def _anima_capabilities() -> RegionalModelCapabilities:
    """Return the fixed model policy used by admission tests."""

    return RegionalModelCapabilities(
        model_family=RegionalModelFamily.ANIMA,
        attention_backend=RegionalAttentionBackend.ANIMA_OBJECT_PATCH,
        attention_topology=RegionalAttentionTopology.SINGLETON_FRAME_SPATIOTEMPORAL,
        latent_layout=RegionalLatentLayout.ANIMA_SINGLE_FRAME_BCTHW,
        spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
        control_gligen_policy=RegionalControlGligenPolicy.REJECT,
        reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
        known_patch_conflicts=(
            RegionalPatchConflict.DIFFUSION_MODEL_WRAPPER,
            RegionalPatchConflict.CROSS_ATTENTION_OBJECT_PATCH,
            RegionalPatchConflict.ATTN2_INPUT_PATCH,
            RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
        ),
    )
