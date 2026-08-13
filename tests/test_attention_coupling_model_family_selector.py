# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify selection from admitted Attention Coupling backend capabilities."""

from __future__ import annotations

import pytest

from simple_syrup.domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalControlGligenPolicy,
    RegionalLatentLayout,
    RegionalModelCapabilities,
    RegionalModelFamily,
    RegionalPatchConflict,
    RegionalReferenceLatentPolicy,
    RegionalSpatialPatchSupport,
)
from simple_syrup.services.anima_attention_coupling_model_family import (
    AnimaAttentionCouplingModelFamily,
)
from simple_syrup.services.attention_coupling_model_family_selector import (
    AttentionCouplingModelFamilySelector,
)
from simple_syrup.services.unet_attention_coupling_model_family import (
    StandardUnetAttentionCouplingModelFamily,
)


@pytest.mark.parametrize(
    ("capabilities", "expected_type"),
    [
        (
            RegionalModelCapabilities(
                RegionalModelFamily.ANIMA,
                RegionalAttentionBackend.ANIMA_OBJECT_PATCH,
                RegionalLatentLayout.ANIMA_SINGLE_FRAME_BCTHW,
                RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
                RegionalControlGligenPolicy.REJECT,
                RegionalReferenceLatentPolicy.REJECT,
                (
                    RegionalPatchConflict.DIFFUSION_MODEL_WRAPPER,
                    RegionalPatchConflict.CROSS_ATTENTION_OBJECT_PATCH,
                    RegionalPatchConflict.ATTN2_INPUT_PATCH,
                    RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
                ),
            ),
            AnimaAttentionCouplingModelFamily,
        ),
        (
            RegionalModelCapabilities(
                RegionalModelFamily.STANDARD_UNET,
                RegionalAttentionBackend.UNET_ATTN2_PATCH,
                RegionalLatentLayout.STANDARD_IMAGE_BCHW,
                RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
                RegionalControlGligenPolicy.REJECT,
                RegionalReferenceLatentPolicy.REJECT,
                (
                    RegionalPatchConflict.ATTN2_INPUT_PATCH,
                    RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
                ),
            ),
            StandardUnetAttentionCouplingModelFamily,
        ),
    ],
)
def test_selector_maps_admitted_backend_to_one_family_adapter(
    capabilities: RegionalModelCapabilities,
    expected_type: type[object],
) -> None:
    """Select without repeating installed model-class detection."""

    selected = AttentionCouplingModelFamilySelector().select(capabilities)

    assert type(selected) is expected_type


def test_selector_rejects_non_capability_input() -> None:
    """Require the exact immutable result of central model admission."""

    with pytest.raises(TypeError, match="requires model capabilities"):
        AttentionCouplingModelFamilySelector().select(object())  # type: ignore[arg-type]
