# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify immutable and internally consistent regional capability values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

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


def test_capability_value_is_frozen() -> None:
    """Prevent admission policy from changing after model detection."""

    capabilities = _standard_capabilities()
    attribute = "model_family"

    with pytest.raises(FrozenInstanceError):
        setattr(capabilities, attribute, RegionalModelFamily.ANIMA)


@pytest.mark.parametrize(
    ("conflicts", "error", "message"),
    [
        (cast(Any, [RegionalPatchConflict.ATTN2_INPUT_PATCH]), TypeError, "tuple"),
        ((), ValueError, "require known patch conflicts"),
        (
            cast(Any, ("attn2_input_patch",)),
            TypeError,
            "must contain RegionalPatchConflict",
        ),
        (
            (
                RegionalPatchConflict.ATTN2_INPUT_PATCH,
                RegionalPatchConflict.ATTN2_INPUT_PATCH,
            ),
            ValueError,
            "must be unique and ordered",
        ),
    ],
)
def test_capability_value_rejects_malformed_conflicts(
    conflicts: tuple[RegionalPatchConflict, ...],
    error: type[Exception],
    message: str,
) -> None:
    """Require immutable typed unique patch-conflict reporting."""

    with pytest.raises(error, match=message):
        _standard_capabilities(conflicts=conflicts)


def test_capability_value_rejects_cross_family_contract() -> None:
    """Prevent a detector from reporting an incoherent topology combination."""

    with pytest.raises(ValueError, match="do not match the model-family contract"):
        RegionalModelCapabilities(
            model_family=RegionalModelFamily.ANIMA,
            attention_backend=RegionalAttentionBackend.UNET_ATTN2_PATCH,
            attention_topology=RegionalAttentionTopology.SEPARATE_IMAGE_AND_CONTEXT,
            latent_layout=RegionalLatentLayout.STANDARD_IMAGE_BCHW,
            spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
            control_gligen_policy=RegionalControlGligenPolicy.REJECT,
            reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
            known_patch_conflicts=(
                RegionalPatchConflict.ATTN2_INPUT_PATCH,
                RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
            ),
        )


def test_capability_value_rejects_untyped_attention_topology() -> None:
    """Prevent free-form topology labels from bypassing family invariants."""

    with pytest.raises(TypeError, match="attention topology"):
        RegionalModelCapabilities(
            model_family=RegionalModelFamily.STANDARD_UNET,
            attention_backend=RegionalAttentionBackend.UNET_ATTN2_PATCH,
            attention_topology=cast(
                Any,
                RegionalAttentionTopology.SEPARATE_IMAGE_AND_CONTEXT.value,
            ),
            latent_layout=RegionalLatentLayout.STANDARD_IMAGE_BCHW,
            spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
            control_gligen_policy=RegionalControlGligenPolicy.REJECT,
            reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
            known_patch_conflicts=(
                RegionalPatchConflict.ATTN2_INPUT_PATCH,
                RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
            ),
        )


def _standard_capabilities(
    *,
    conflicts: tuple[RegionalPatchConflict, ...] = (
        RegionalPatchConflict.ATTN2_INPUT_PATCH,
        RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
    ),
) -> RegionalModelCapabilities:
    """Return one complete standard topology value."""

    return RegionalModelCapabilities(
        model_family=RegionalModelFamily.STANDARD_UNET,
        attention_backend=RegionalAttentionBackend.UNET_ATTN2_PATCH,
        attention_topology=RegionalAttentionTopology.SEPARATE_IMAGE_AND_CONTEXT,
        latent_layout=RegionalLatentLayout.STANDARD_IMAGE_BCHW,
        spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
        control_gligen_policy=RegionalControlGligenPolicy.REJECT,
        reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
        known_patch_conflicts=conflicts,
    )
