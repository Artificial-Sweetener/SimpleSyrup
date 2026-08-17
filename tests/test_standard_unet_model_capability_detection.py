# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify graph-driven standard-UNet regional capability detection."""

from __future__ import annotations

from typing import TypedDict, cast

import comfy.latent_formats
import comfy.model_base
import pytest
import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from regional_model_capability_test_values import (
    AlternateImageLatent,
    TemporalLatent,
    patcher,
    standard_unet_graph,
)

from simple_syrup.domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalAttentionTopology,
    RegionalLatentLayout,
    RegionalModelFamily,
)
from simple_syrup.runtime.standard_unet_model_capability import (
    StandardUnetModelCapabilityDetector,
)


class _GraphOptions(TypedDict, total=False):
    """Describe supported graph mutations for rejection characterization."""

    include_spatial_transformer: bool
    disable_self_attention: bool
    missing_cross_attention: bool
    video_transformer: bool
    video_resblock: bool


@pytest.mark.parametrize(
    "base_type",
    [
        comfy.model_base.BaseModel,
        comfy.model_base.SDXL,
        comfy.model_base.SDXLRefiner,
    ],
)
def test_compatible_standard_unet_graphs_report_complete_capabilities(
    base_type: type[torch.nn.Module],
) -> None:
    """Admit standard wrappers from their shared live topology contract."""

    model = patcher(
        base_type=base_type,
        diffusion_model=standard_unet_graph(),
        latent_format=comfy.latent_formats.SD15(),
    )

    capabilities = StandardUnetModelCapabilityDetector().detect(model)

    assert capabilities is not None
    assert capabilities.model_family is RegionalModelFamily.STANDARD_UNET
    assert capabilities.attention_backend is RegionalAttentionBackend.UNET_ATTN2_PATCH
    assert capabilities.attention_topology is (
        RegionalAttentionTopology.SEPARATE_IMAGE_AND_CONTEXT
    )
    assert capabilities.latent_layout is RegionalLatentLayout.STANDARD_IMAGE_BCHW


def test_compatible_wrapper_diffusion_and_latent_subclasses_are_admitted() -> None:
    """Derive support from capabilities instead of exact wrapper/latent names."""

    base_subclass = cast(
        type[torch.nn.Module],
        type("CompatibleBaseModel", (comfy.model_base.BaseModel,), {}),
    )
    diffusion_subclass = cast(
        type[UNetModel],
        type("CompatibleUNetModel", (UNetModel,), {}),
    )
    model = patcher(
        base_type=base_subclass,
        diffusion_model=standard_unet_graph(diffusion_type=diffusion_subclass),
        latent_format=AlternateImageLatent(),
    )

    assert StandardUnetModelCapabilityDetector().detect(model) is not None


def test_temporal_latent_layout_fails_with_an_actionable_reason() -> None:
    """Reject temporal sampling before standard image attention mutation."""

    model = patcher(
        base_type=comfy.model_base.BaseModel,
        diffusion_model=standard_unet_graph(),
        latent_format=TemporalLatent(),
    )

    with pytest.raises(ValueError, match="two-dimensional image latent"):
        StandardUnetModelCapabilityDetector().detect(model)


@pytest.mark.parametrize(
    ("graph_options", "message"),
    [
        ({"include_spatial_transformer": False}, "SpatialTransformer"),
        ({"video_transformer": True}, "SpatialVideoTransformer"),
        ({"video_resblock": True}, "VideoResBlock"),
        ({"missing_cross_attention": True}, "attn2 cross-attention"),
        ({"disable_self_attention": True}, "attn1 self-attention"),
    ],
)
def test_unsupported_attention_topologies_fail_with_exact_reasons(
    graph_options: _GraphOptions,
    message: str,
) -> None:
    """Reject every graph whose token or temporal roles exceed the backend."""

    model = patcher(
        base_type=comfy.model_base.BaseModel,
        diffusion_model=standard_unet_graph(**graph_options),
        latent_format=AlternateImageLatent(),
    )

    with pytest.raises(ValueError, match=message):
        StandardUnetModelCapabilityDetector().detect(model)


def test_non_unet_diffusion_graph_is_not_claimed() -> None:
    """Return no match when the standard backend cannot own the graph."""

    model = patcher(
        base_type=comfy.model_base.BaseModel,
        diffusion_model=torch.nn.Linear(4, 4),
        latent_format=AlternateImageLatent(),
    )

    assert StandardUnetModelCapabilityDetector().detect(model) is None
