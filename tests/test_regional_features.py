# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify immutable regional feature requests, capabilities, and admissions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from simple_syrup.domain.regional_features import (
    EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    RegionalCapabilityAdmission,
    RegionalFeature,
    RegionalFeatureRequest,
    RegionalSamplerCapabilities,
)


def test_request_adds_features_without_mutating_the_source() -> None:
    """Build a new immutable request while preserving the empty request."""

    source = RegionalFeatureRequest()
    requested = source.with_feature(RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING)

    assert source.features == frozenset()
    assert requested.features == frozenset(
        {RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING}
    )


def test_complete_admission_reports_typed_feature_support() -> None:
    """Expose only features present in the completely admitted request."""

    request = RegionalFeatureRequest(
        frozenset({RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING})
    )
    admission = RegionalCapabilityAdmission(request, request.features, None)

    assert admission.supports(RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING)
    assert not admission.supports(RegionalFeature.ATTENTION_COUPLING)
    assert not EMPTY_REGIONAL_CAPABILITY_ADMISSION.admitted_features


def test_feature_values_are_frozen() -> None:
    """Prevent feature intent from changing after request construction."""

    request = RegionalFeatureRequest()
    attribute = "features"

    with pytest.raises(FrozenInstanceError):
        setattr(request, attribute, frozenset({RegionalFeature.CONTROL}))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: RegionalFeatureRequest(cast(Any, {RegionalFeature.CONTROL})),
            "immutable frozenset",
        ),
        (
            lambda: RegionalSamplerCapabilities(cast(Any, frozenset({"control"}))),
            "contain RegionalFeature values",
        ),
    ],
)
def test_feature_sets_reject_mutable_or_untyped_values(
    factory: Any,
    message: str,
) -> None:
    """Reject feature collections that can drift or bypass typed policy."""

    with pytest.raises(TypeError, match=message):
        factory()


def test_admission_rejects_partial_feature_results() -> None:
    """Forbid silently dropping a requested feature during admission."""

    request = RegionalFeatureRequest(
        frozenset({RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING})
    )

    with pytest.raises(ValueError, match="cannot be partial"):
        RegionalCapabilityAdmission(request, frozenset(), None)


def test_model_dependent_admission_requires_model_capabilities() -> None:
    """Forbid attention or patch admission without defensive model detection."""

    request = RegionalFeatureRequest(frozenset({RegionalFeature.ATTENTION_COUPLING}))

    with pytest.raises(ValueError, match="require model capabilities"):
        RegionalCapabilityAdmission(request, request.features, None)
