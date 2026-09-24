# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify model-neutral regional activation geometry invariants."""

from __future__ import annotations

import pytest

from simple_syrup.domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationGeometry,
    RegionalActivationLayout,
    RegionalTemporalOwnership,
)
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)


def test_batch_alignment_composes_cfg_latent_and_view_major_counts() -> None:
    """Use existing layout evidence to derive one exact invocation batch."""

    layout = SpatialBatchLayout(
        canvas_width=8,
        canvas_height=6,
        views=(
            SpatialView(SpatialViewKind.TILE, 0, 0, 4, 6, 4, 6),
            SpatialView(SpatialViewKind.TILE, 4, 0, 4, 6, 4, 6),
        ),
        input_batch_size=4,
    )

    alignment = RegionalActivationBatchAlignment(
        latent_batch_size=2,
        chunk_count=2,
        spatial_layout=layout,
    )

    assert alignment.base_batch_size == 4
    assert alignment.invocation_batch_size == 8


@pytest.mark.parametrize(
    ("layout", "shape", "height", "width", "mask_shape"),
    [
        (
            RegionalActivationLayout.DIRECT_CONVOLUTION_1D,
            (4, 8, 12),
            1,
            12,
            (3, 4, 1, 12),
        ),
        (
            RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
            (4, 8, 6, 12),
            6,
            12,
            (3, 4, 1, 6, 12),
        ),
        (
            RegionalActivationLayout.FLATTENED_SPATIAL_TOKENS,
            (4, 72, 8),
            6,
            12,
            (3, 4, 72, 1),
        ),
        (
            RegionalActivationLayout.CONSUMER_SPATIALIZED,
            (4, 72, 8),
            6,
            12,
            (3, 4, 72, 1),
        ),
        (
            RegionalActivationLayout.BRANCH_TOKENS,
            (4, 77, 8),
            1,
            77,
            (3, 4, 77, 1),
        ),
    ],
)
def test_geometry_admits_explicit_image_layouts(
    layout: RegionalActivationLayout,
    shape: tuple[int, ...],
    height: int,
    width: int,
    mask_shape: tuple[int, ...],
) -> None:
    """Admit exact image layouts without inferring spatial dimensions."""

    feature_axis = 1 if "convolution" in layout.value else 2
    geometry = RegionalActivationGeometry(
        layout=layout,
        invocation_shape=shape,
        feature_axis=feature_axis,
        spatial_height=height,
        spatial_width=width,
        batch_alignment=RegionalActivationBatchAlignment(2, 2),
    )

    assert geometry.broadcast_mask_shape(3) == mask_shape
    assert geometry.temporal_size is None


def test_geometry_admits_conv3d_only_with_explicit_temporal_ownership() -> None:
    """Require the caller to own repetition of authored masks over time."""

    geometry = RegionalActivationGeometry(
        layout=RegionalActivationLayout.DIRECT_CONVOLUTION_3D,
        invocation_shape=(2, 8, 3, 6, 12),
        feature_axis=1,
        spatial_height=6,
        spatial_width=12,
        temporal_axis=2,
        temporal_ownership=RegionalTemporalOwnership.REPEAT_SPATIAL_MASK,
        batch_alignment=RegionalActivationBatchAlignment(2, 1),
    )

    assert geometry.temporal_size == 3
    assert geometry.broadcast_mask_shape(4) == (4, 2, 1, 3, 6, 12)


def test_geometry_admits_internal_grid_below_published_view_resolution() -> None:
    """Keep the view as crop evidence while accepting observed UNet downsampling."""

    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
        (2, 8, 3, 6),
        1,
        3,
        6,
        RegionalActivationBatchAlignment(
            2,
            1,
            _full_layout(input_batch_size=2),
        ),
    )

    assert geometry.invocation_shape == (2, 8, 3, 6)
    assert geometry.batch_alignment.spatial_layout is not None
    assert geometry.batch_alignment.spatial_layout.views[0].model_width == 12


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("layout-batch", "input batch"),
        ("invocation-batch", "leading batch"),
        ("token-count", "expected shape"),
        ("conv-height", "height one"),
        ("conv3d-ownership", "explicit repeated"),
        ("temporal-on-image", "cannot declare a temporal axis"),
    ],
)
def test_geometry_rejects_ambiguous_or_divergent_layouts(
    case: str,
    message: str,
) -> None:
    """Reject every geometry claim not proven by explicit invocation evidence."""

    with pytest.raises(ValueError, match=message):
        _invalid_case(case)


def _invalid_case(case: str) -> object:
    """Construct one invalid value after parametrized test collection."""

    if case == "layout-batch":
        return RegionalActivationBatchAlignment(
            2,
            2,
            _full_layout(input_batch_size=2),
        )
    if case == "invocation-batch":
        return RegionalActivationGeometry(
            RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
            (3, 8, 6, 12),
            1,
            6,
            12,
            RegionalActivationBatchAlignment(2, 2),
        )
    if case == "token-count":
        return RegionalActivationGeometry(
            RegionalActivationLayout.FLATTENED_SPATIAL_TOKENS,
            (4, 71, 8),
            2,
            6,
            12,
            RegionalActivationBatchAlignment(2, 2),
        )
    if case == "conv-height":
        return RegionalActivationGeometry(
            RegionalActivationLayout.DIRECT_CONVOLUTION_1D,
            (4, 8, 12),
            1,
            2,
            12,
            RegionalActivationBatchAlignment(2, 2),
        )
    if case == "conv3d-ownership":
        return RegionalActivationGeometry(
            RegionalActivationLayout.DIRECT_CONVOLUTION_3D,
            (4, 8, 3, 6, 12),
            1,
            6,
            12,
            RegionalActivationBatchAlignment(2, 2),
            temporal_axis=2,
        )
    if case == "temporal-on-image":
        return RegionalActivationGeometry(
            RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
            (4, 8, 6, 12),
            1,
            6,
            12,
            RegionalActivationBatchAlignment(2, 2),
            temporal_axis=2,
        )
    raise AssertionError(f"Unknown regional activation invalid case: {case}")


def _full_layout(*, input_batch_size: int) -> SpatialBatchLayout:
    """Return one full-canvas layout with an explicit input batch."""

    return SpatialBatchLayout(
        12,
        6,
        (SpatialView(SpatialViewKind.FULL, 0, 0, 12, 6, 12, 6),),
        input_batch_size,
    )
