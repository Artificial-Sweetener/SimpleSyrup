# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify backend-neutral regional query-mask batching."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
)
from simple_syrup.runtime.regional_attention_diagnostic_values import (
    RegionalAttentionQueryGeometry,
)
from simple_syrup.runtime.regional_attention_query_masks import (
    RegionalAttentionQueryMaskProjector,
)


def test_implicit_full_query_masks_repeat_over_cfg_chunks() -> None:
    """Expand one latent-batch layout into exact chunk-major query masks."""

    bank = _bank()
    source = bank.conditioning_masks.clone()
    layout = _full_layout(input_batch_size=2)

    result = RegionalAttentionQueryMaskProjector().project(
        bank=bank,
        query_geometry=RegionalAttentionQueryGeometry(4, 1, 2, 4, None),
        layout=layout,
        form=RegionalMaskForm.CONDITIONING,
        mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result.masks.shape == (2, 4, 2, 4)
    assert torch.equal(result.masks[:, 0], bank.conditioning_masks)
    assert torch.equal(result.masks[:, 1], bank.conditioning_masks)
    assert torch.equal(result.masks[:, 2], bank.conditioning_masks)
    assert torch.equal(result.masks[:, 3], bank.conditioning_masks)
    assert torch.equal(result.flattened, result.masks.flatten(start_dim=2))
    assert torch.equal(bank.conditioning_masks, source)


def test_explicit_tile_query_masks_preserve_view_major_batch_order() -> None:
    """Crop each view before repeating its source batch contiguously."""

    bank = _bank()
    layout = _tile_layout()

    result = RegionalAttentionQueryMaskProjector().project(
        bank=bank,
        query_geometry=RegionalAttentionQueryGeometry(4, 1, 1, 1, layout),
        layout=layout,
        form=RegionalMaskForm.CONDITIONING,
        mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    expected_left = torch.tensor([0.75, 0.25], dtype=torch.float64)
    expected_right = torch.tensor([0.25, 0.75], dtype=torch.float64)
    assert result.masks.shape == (2, 4, 1, 1)
    assert torch.equal(result.masks[:, 0, 0, 0], expected_left)
    assert torch.equal(result.masks[:, 1, 0, 0], expected_left)
    assert torch.equal(result.masks[:, 2, 0, 0], expected_right)
    assert torch.equal(result.masks[:, 3, 0, 0], expected_right)
    assert result.masks.dtype is torch.float64


def test_query_masks_reject_batch_and_layout_authority_drift() -> None:
    """Fail before returning masks when call-local geometry is inconsistent."""

    bank = _bank()
    projector = RegionalAttentionQueryMaskProjector()

    with pytest.raises(ValueError, match="divide over its full layout"):
        projector.project(
            bank=bank,
            query_geometry=RegionalAttentionQueryGeometry(3, 1, 2, 4, None),
            layout=_full_layout(input_batch_size=2),
            form=RegionalMaskForm.CONDITIONING,
            mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    layout = _tile_layout()
    with pytest.raises(ValueError, match="published geometry authority"):
        projector.project(
            bank=bank,
            query_geometry=RegionalAttentionQueryGeometry(4, 1, 1, 1, layout),
            layout=_tile_layout(),
            form=RegionalMaskForm.CONDITIONING,
            mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def _bank() -> RegionalMaskBank:
    """Return two complementary soft masks over a rectangular canvas."""

    masks = torch.tensor(
        [
            [[1.0, 0.5, 0.5, 0.0], [1.0, 0.5, 0.5, 0.0]],
            [[0.0, 0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0]],
        ]
    )
    return RegionalMaskBank(masks, masks.clone(), 4, 2)


def _full_layout(*, input_batch_size: int) -> SpatialBatchLayout:
    """Return one full-canvas source layout."""

    return SpatialBatchLayout(
        4,
        2,
        (SpatialView(SpatialViewKind.FULL, 0, 0, 4, 2, 4, 2),),
        input_batch_size,
    )


def _tile_layout() -> SpatialBatchLayout:
    """Return two view-major half-canvas tiles."""

    return SpatialBatchLayout(
        4,
        2,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 2, 2, 2, 2),
            SpatialView(SpatialViewKind.TILE, 2, 0, 2, 2, 2, 2),
        ),
        2,
    )
