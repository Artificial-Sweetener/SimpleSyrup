# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify installed-role discovery for standard-UNet regional Linear targets."""

from __future__ import annotations

from comfy.ldm.modules.attention import SpatialTransformer
from torch import nn

from simple_syrup.runtime.regional_lora.standard_unet_target_capabilities import (
    StandardUnetTargetCapabilityClassifier,
)
from simple_syrup.runtime.regional_lora.target_binding import (
    BoundRegionalLoraSpatialCapability,
)


class _Graph(nn.Module):
    """Expose one installed Comfy transformer and one misleading unrelated tree."""

    def __init__(self, *, disable_self_attn: bool = False) -> None:
        """Build a small standard attention graph on CPU."""

        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.spatial = SpatialTransformer(
            32,
            4,
            8,
            depth=1,
            context_dim=64,
            disable_self_attn=disable_self_attn,
            use_linear=True,
            use_checkpoint=False,
        )
        self.diffusion_model.attn2 = nn.Module()
        self.diffusion_model.attn2.to_k = nn.Linear(64, 32, bias=False)


def test_classifier_preserves_distinct_image_and_context_token_roles() -> None:
    """Classify every standard transformer Linear from its installed owner."""

    result = StandardUnetTargetCapabilityClassifier().classify(_Graph())
    prefix = "diffusion_model.spatial.transformer_blocks.0"

    assert result.unsupported_blocks == ()
    assert len(result.linear_roles) == 12
    assert result.linear_roles[f"{prefix}.attn1.to_k.weight"] is (
        BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS
    )
    assert result.linear_roles[f"{prefix}.attn2.to_q.weight"] is (
        BoundRegionalLoraSpatialCapability.PACKED_IMAGE_TOKENS
    )
    assert result.linear_roles[f"{prefix}.attn2.to_k.weight"] is (
        BoundRegionalLoraSpatialCapability.PACKED_CONTEXT_TOKENS
    )
    assert "diffusion_model.attn2.to_k.weight" not in result.linear_roles


def test_classifier_rejects_cross_attention_in_the_attn1_slot() -> None:
    """Refuse a block whose activation roles exceed the standard paired path."""

    result = StandardUnetTargetCapabilityClassifier().classify(
        _Graph(disable_self_attn=True)
    )

    assert result.unsupported_blocks == (
        "diffusion_model.spatial.transformer_blocks.0",
    )
    assert tuple(result.linear_roles) == (
        "diffusion_model.spatial.proj_in.weight",
        "diffusion_model.spatial.proj_out.weight",
    )
