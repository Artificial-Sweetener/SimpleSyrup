# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify explicit runtime providers for regional activation geometry."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationLayout,
    RegionalTemporalOwnership,
)
from simple_syrup.runtime.regional_lora.activation_geometry_providers import (
    DirectConvolutionGeometryProvider,
    SpatialTokenGeometryProvider,
)


@pytest.mark.parametrize(
    ("shape", "layout", "height", "width"),
    [
        ((2, 4, 9), RegionalActivationLayout.DIRECT_CONVOLUTION_1D, 1, 9),
        ((2, 4, 5, 9), RegionalActivationLayout.DIRECT_CONVOLUTION_2D, 5, 9),
    ],
)
def test_direct_provider_reads_exact_image_tensor_axes(
    shape: tuple[int, ...],
    layout: RegionalActivationLayout,
    height: int,
    width: int,
) -> None:
    """Describe only the conventional dimensions observed on the tensor."""

    geometry = DirectConvolutionGeometryProvider().resolve(
        torch.ones(shape),
        batch_alignment=RegionalActivationBatchAlignment(2, 1),
    )

    assert geometry.layout is layout
    assert geometry.spatial_height == height
    assert geometry.spatial_width == width


def test_direct_provider_requires_explicit_conv3d_temporal_ownership() -> None:
    """Admit temporal convolution only when authored-mask repetition is declared."""

    activation = torch.ones((2, 4, 3, 5, 9))
    with pytest.raises(ValueError, match="explicit repeated"):
        DirectConvolutionGeometryProvider().resolve(
            activation,
            batch_alignment=RegionalActivationBatchAlignment(2, 1),
        )

    geometry = DirectConvolutionGeometryProvider(
        temporal_ownership=RegionalTemporalOwnership.REPEAT_SPATIAL_MASK
    ).resolve(
        activation,
        batch_alignment=RegionalActivationBatchAlignment(2, 1),
    )

    assert geometry.temporal_size == 3


@pytest.mark.parametrize("consumer_spatialized", [False, True])
def test_token_provider_requires_published_rectangular_grid(
    consumer_spatialized: bool,
) -> None:
    """Accept a non-square token grid only from explicit H/W publication."""

    geometry = SpatialTokenGeometryProvider(
        spatial_height=3,
        spatial_width=8,
        consumer_spatialized=consumer_spatialized,
    ).resolve(
        torch.ones((2, 24, 6)),
        batch_alignment=RegionalActivationBatchAlignment(2, 1),
    )

    expected = (
        RegionalActivationLayout.CONSUMER_SPATIALIZED
        if consumer_spatialized
        else RegionalActivationLayout.FLATTENED_SPATIAL_TOKENS
    )
    assert geometry.layout is expected


def test_token_provider_does_not_infer_grid_from_token_count() -> None:
    """Reject a mismatched published grid even when token count has square factors."""

    with pytest.raises(ValueError, match="expected shape"):
        SpatialTokenGeometryProvider(spatial_height=4, spatial_width=4).resolve(
            torch.ones((2, 24, 6)),
            batch_alignment=RegionalActivationBatchAlignment(2, 1),
        )


@pytest.mark.parametrize(
    "activation",
    [
        torch.ones((1, 4, 5, 5)),
        torch.ones((2, 4, 5, 5), dtype=torch.int64),
        torch.ones((2, 4)),
    ],
)
def test_direct_provider_rejects_batch_dtype_and_rank_mismatch(
    activation: torch.Tensor,
) -> None:
    """Fail before publishing incomplete or incompatible invocation geometry."""

    with pytest.raises((TypeError, ValueError)):
        DirectConvolutionGeometryProvider().resolve(
            activation,
            batch_alignment=RegionalActivationBatchAlignment(2, 1),
        )
