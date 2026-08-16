# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact standard-UNet attn2 query-geometry resolution."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.attention_coupling.unet_attention_geometry import (
    StandardUnetAttentionGeometryResolver,
)
from simple_syrup.runtime.spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


def test_unet_geometry_resolves_rectangular_full_attention_resolution() -> None:
    """Synthesize one full layout while retaining exact internal H/W."""

    geometry = StandardUnetAttentionGeometryResolver().resolve(
        torch.zeros(2, 24, 32),
        _contexts(),
        _mask_bank(),
        _options(
            activations_shape=[2, 32, 4, 6],
            original_shape=[2, 4, 8, 12],
        ),
    )

    assert geometry.query.input_batch_size == 2
    assert geometry.query.query_height == 4
    assert geometry.query.query_width == 6
    assert geometry.query.query_token_count == 24
    assert geometry.query.spatial_layout is None
    assert geometry.layout.views[0].kind is SpatialViewKind.FULL
    assert geometry.layout.input_batch_size == 1
    assert (geometry.original_height, geometry.original_width) == (8, 12)
    assert geometry.block == ("input", 4)
    assert geometry.block_index == 0
    assert geometry.transformer_index == 2


def test_unet_geometry_retains_exact_published_tile_layout_identity() -> None:
    """Use the call's view-major layout without rebuilding equal geometry."""

    layout = _tile_layout()
    options = _options(
        activations_shape=[2, 32, 2, 3],
        original_shape=[2, 4, 4, 6],
    )
    options[SIMPLE_SYRUP_TRANSFORMER_NAMESPACE] = {SPATIAL_BATCH_LAYOUT_KEY: layout}

    geometry = StandardUnetAttentionGeometryResolver().resolve(
        torch.zeros(2, 6, 32),
        _contexts(),
        _mask_bank(),
        options,
    )

    assert geometry.query.spatial_layout is layout
    assert geometry.layout is layout
    assert geometry.query.query_height == 2
    assert geometry.query.query_width == 3


@pytest.mark.parametrize(
    ("query", "changes", "message"),
    [
        (torch.zeros(2, 23, 32), {}, "token count"),
        (torch.zeros(1, 24, 32), {}, "activation batch"),
        (torch.zeros(2, 24, 32), {"block": ("foreign", 0)}, "unsupported"),
        (torch.zeros(2, 24, 32), {"block_index": -1}, "non-negative"),
        (torch.zeros(2, 24, 32), {"transformer_index": True}, "integer"),
    ],
)
def test_unet_geometry_rejects_callback_metadata_drift(
    query: torch.Tensor,
    changes: dict[str, object],
    message: str,
) -> None:
    """Fail closed when installed callback metadata loses its contract."""

    options = _options(
        activations_shape=[2, 32, 4, 6],
        original_shape=[2, 4, 8, 12],
    )
    options.update(changes)

    with pytest.raises((TypeError, ValueError), match=message):
        StandardUnetAttentionGeometryResolver().resolve(
            query,
            _contexts(),
            _mask_bank(),
            options,
        )


def test_unet_geometry_rejects_published_layout_model_shape_drift() -> None:
    """Reject view metadata that cannot describe the active UNet tensor."""

    layout = _tile_layout()
    options = _options(
        activations_shape=[2, 32, 2, 3],
        original_shape=[2, 4, 8, 6],
    )
    options[SIMPLE_SYRUP_TRANSFORMER_NAMESPACE] = {SPATIAL_BATCH_LAYOUT_KEY: layout}

    with pytest.raises(ValueError, match="layout model view"):
        StandardUnetAttentionGeometryResolver().resolve(
            torch.zeros(2, 6, 32),
            _contexts(),
            _mask_bank(),
            options,
        )


def _options(
    *,
    activations_shape: list[int],
    original_shape: list[int],
) -> dict[str, Any]:
    """Return installed callback metadata for one input transformer."""

    return {
        "activations_shape": activations_shape,
        "original_shape": original_shape,
        "block": ("input", 4),
        "block_index": 0,
        "transformer_index": 2,
    }


def _mask_bank() -> RegionalMaskBank:
    """Return one canonical 12-by-8 regional mask bank."""

    masks = torch.ones(1, 8, 12)
    return RegionalMaskBank(masks, masks.clone(), 12, 8)


def _contexts() -> BatchedRegionalAttentionContexts:
    """Return one positive and one negative context at latent batch one."""

    return BatchedRegionalAttentionContexts(
        1,
        (
            RegionalAttentionChunkBatch(0, RegionalAttentionBranch.NEGATIVE, 0, 1),
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.POSITIVE, 1, 2),
        ),
        torch.zeros(2, 3, 16),
        (
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, torch.ones(2, 3, 16), (1.0, 1.0)),),
            ),
        ),
    )


def _tile_layout() -> SpatialBatchLayout:
    """Return two view-major tiles evaluated at 6-by-4 model shape."""

    return SpatialBatchLayout(
        12,
        8,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 6, 8, 6, 4),
            SpatialView(SpatialViewKind.TILE, 6, 0, 6, 8, 6, 4),
        ),
        1,
    )
