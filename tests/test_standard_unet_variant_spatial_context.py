# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify standard-UNet variant spatial execution identity."""

from __future__ import annotations

import pytest

from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_spatial_context import (
    STANDARD_UNET_VARIANT_SPATIAL_CONTEXT,
)
from simple_syrup.runtime.spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


def test_absent_layout_identifies_full_canvas_execution() -> None:
    """Treat the native model call as full-canvas spatial evidence."""

    assert STANDARD_UNET_VARIANT_SPATIAL_CONTEXT.modes({}) == ("full",)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (SpatialViewKind.TILE, ("tile",)),
        (SpatialViewKind.CONTEXTUAL_GLOBAL, ("contextual_global",)),
    ],
)
def test_published_layout_identifies_its_exact_view_kind(
    kind: SpatialViewKind,
    expected: tuple[str, ...],
) -> None:
    """Report the authoritative spatial kind supplied by the sampler."""

    layout = _layout(kind)
    options: dict[str, object] = {
        SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {
            SPATIAL_BATCH_LAYOUT_KEY: layout,
        }
    }

    assert STANDARD_UNET_VARIANT_SPATIAL_CONTEXT.layout(options) is layout
    assert STANDARD_UNET_VARIANT_SPATIAL_CONTEXT.modes(options) == expected


@pytest.mark.parametrize(
    "options",
    [
        {SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: []},
        {
            SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {
                SPATIAL_BATCH_LAYOUT_KEY: object(),
            }
        },
    ],
)
def test_malformed_spatial_metadata_fails_closed(options: dict[str, object]) -> None:
    """Reject malformed namespaces and layouts before evidence is emitted."""

    with pytest.raises(TypeError, match="spatial"):
        STANDARD_UNET_VARIANT_SPATIAL_CONTEXT.layout(options)


def _layout(kind: SpatialViewKind) -> SpatialBatchLayout:
    """Return one valid anonymous layout for the selected spatial kind."""

    canvas = 128
    source = canvas if kind is SpatialViewKind.CONTEXTUAL_GLOBAL else 64
    return SpatialBatchLayout(
        canvas,
        canvas,
        (
            SpatialView(
                kind,
                0,
                0,
                source,
                source,
                64,
                64,
            ),
        ),
        2,
    )
