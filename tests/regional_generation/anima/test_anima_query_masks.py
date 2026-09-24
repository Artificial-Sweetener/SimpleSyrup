# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove canonical tile masks reach exact Anima query-grid coordinates."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

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
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import (
    AnimaQueryMaskProjector,
)


def test_tile_matrix_projects_contained_boundary_edge_and_uncovered_masks() -> None:
    """Crop every ordered tile before projecting its two-region query grid."""

    masks = torch.zeros((2, 8, 10), dtype=torch.float32)
    masks[0, 0:4, 0:4] = 1.0
    masks[1, 0:4, 6:10] = 1.0
    bank = _bank(masks)
    views = (
        _tile(x=0, y=0),
        _tile(x=6, y=0),
        _tile(x=3, y=0),
        _tile(x=6, y=4),
        _tile(x=2, y=4),
    )
    layout = SpatialBatchLayout(10, 8, views, input_batch_size=1)

    projected = AnimaQueryMaskProjector().project(
        bank=bank,
        geometry=_geometry(
            layout=layout,
            input_batch_size=5,
            activation_height=4,
            activation_width=4,
            query_height=2,
            query_width=2,
        ),
        latent_batch_size=1,
        form=RegionalMaskForm.CONDITIONING,
        mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.5, 0.0], [0.5, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[0.0, 0.5], [0.0, 0.5]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(projected.masks, expected)
    torch.testing.assert_close(projected.flattened, expected.flatten(start_dim=2))


@pytest.mark.parametrize("tile_batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("chunk_count", [1, 2], ids=("positive-only", "ordinary-cfg"))
def test_masks_follow_view_chunk_and_latent_model_batch_order(
    tile_batch_size: int,
    chunk_count: int,
) -> None:
    """Repeat each tile mask over two CFG chunks and two latent samples."""

    canvas_width = tile_batch_size * 2
    masks = torch.zeros((1, 2, canvas_width), dtype=torch.float32)
    for view_index in range(tile_batch_size):
        masks[:, :, view_index * 2 : (view_index + 1) * 2] = (
            view_index + 1
        ) / tile_batch_size
    bank = _bank(masks)
    layout = SpatialBatchLayout(
        canvas_width,
        2,
        tuple(
            SpatialView(SpatialViewKind.TILE, index * 2, 0, 2, 2, 2, 2)
            for index in range(tile_batch_size)
        ),
        input_batch_size=chunk_count * 2,
    )

    projected = AnimaQueryMaskProjector().project(
        bank=bank,
        geometry=_geometry(
            layout=layout,
            input_batch_size=tile_batch_size * chunk_count * 2,
            activation_height=2,
            activation_width=2,
            query_height=2,
            query_width=2,
        ),
        latent_batch_size=2,
        form=RegionalMaskForm.CONDITIONING,
        mode=RegionalMaskProjectionMode.HARD_PRESERVING,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_values = torch.tensor(
        [
            (view_index + 1) / tile_batch_size
            for view_index in range(tile_batch_size)
            for _chunk_index in range(chunk_count)
            for _latent_index in range(2)
        ]
    )
    torch.testing.assert_close(projected.masks[0, :, 0, 0, 0], expected_values)


@pytest.mark.parametrize(
    ("source_width", "source_height", "model_size", "query_size"),
    [
        (4, 4, 4, 2),
        (2, 2, 8, 4),
    ],
)
def test_tile_crop_projects_directly_to_smaller_or_larger_query_grid(
    source_width: int,
    source_height: int,
    model_size: int,
    query_size: int,
) -> None:
    """Resize only the selected source rectangle to the active query H/W."""

    masks = torch.arange(48, dtype=torch.float32).reshape((1, 6, 8)) / 47.0
    bank = _bank(masks)
    view = SpatialView(
        SpatialViewKind.TILE,
        source_x=2,
        source_y=1,
        source_width=source_width,
        source_height=source_height,
        model_width=model_size,
        model_height=model_size,
    )
    layout = SpatialBatchLayout(8, 6, (view,), input_batch_size=1)

    projected = AnimaQueryMaskProjector().project(
        bank=bank,
        geometry=_geometry(
            layout=layout,
            input_batch_size=1,
            activation_height=model_size,
            activation_width=model_size,
            query_height=query_size,
            query_width=query_size,
        ),
        latent_batch_size=1,
        form=RegionalMaskForm.CONDITIONING,
        mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    cropped = masks[:, 1 : 1 + source_height, 2 : 2 + source_width]
    if query_size <= min(source_height, source_width):
        expected = functional.interpolate(
            cropped.unsqueeze(1),
            size=(query_size, query_size),
            mode="area",
        ).squeeze(1)
    else:
        expected = functional.interpolate(
            cropped.unsqueeze(1),
            size=(query_size, query_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    torch.testing.assert_close(
        projected.masks,
        expected.to(dtype=torch.float64).unsqueeze(1).unsqueeze(2),
    )


def _tile(*, x: int, y: int) -> SpatialView:
    """Build one four-by-four identity-sized tile."""

    return SpatialView(SpatialViewKind.TILE, x, y, 4, 4, 4, 4)


def _bank(masks: torch.Tensor) -> RegionalMaskBank:
    """Build one canonical bank with identical planning and conditioning forms."""

    return RegionalMaskBank(
        planning_masks=masks.clone(),
        conditioning_masks=masks.clone(),
        canvas_width=int(masks.shape[-1]),
        canvas_height=int(masks.shape[-2]),
    )


def _geometry(
    *,
    layout: SpatialBatchLayout,
    input_batch_size: int,
    activation_height: int,
    activation_width: int,
    query_height: int,
    query_width: int,
) -> AnimaActivationGeometry:
    """Build exact active geometry for one tiled Anima invocation."""

    return AnimaActivationGeometry(
        input_batch_size=input_batch_size,
        activation_time=1,
        activation_height=activation_height,
        activation_width=activation_width,
        patch_temporal=1,
        patch_spatial=max(1, activation_height // query_height),
        query_time=1,
        query_height=query_height,
        query_width=query_width,
        spatial_layout=layout,
    )
