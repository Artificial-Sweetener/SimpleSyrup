# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for tile SEGS domain construction."""

from __future__ import annotations

import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion
from simple_syrup.domain.tile_segs import TileSEGSBuilder, TileSEGSControls


def test_small_image_creates_one_clamped_tile() -> None:
    """A bbox larger than the image clamps to one full-image tile."""

    segs = TileSEGSBuilder().build(_image(6, 6), _controls(bbox_size=64))

    assert segs[0] == (6, 6)
    assert len(segs[1]) == 1
    assert segs[1][0].bbox == BoundingBox(0, 0, 6, 6)
    assert segs[1][0].crop_region == CropRegion(0, 0, 6, 6)


def test_tiles_are_row_major_and_cover_edges() -> None:
    """Tile generation is deterministic and reaches image boundaries."""

    segs = TileSEGSBuilder().build(
        _image(6, 10),
        _controls(bbox_size=4, min_overlap=1),
    )

    assert [segment.bbox for segment in segs[1]] == [
        BoundingBox(0, 0, 4, 4),
        BoundingBox(2, 0, 6, 4),
        BoundingBox(4, 0, 8, 4),
        BoundingBox(6, 0, 10, 4),
        BoundingBox(0, 2, 4, 6),
        BoundingBox(2, 2, 6, 6),
        BoundingBox(4, 2, 8, 6),
        BoundingBox(6, 2, 10, 6),
    ]
    assert [segment.label for segment in segs[1]] == [
        "tile_001",
        "tile_002",
        "tile_003",
        "tile_004",
        "tile_005",
        "tile_006",
        "tile_007",
        "tile_008",
    ]


def test_irregular_reuse_mode_reuses_mask_pattern() -> None:
    """Reuse modes apply the same generated mask pattern to each tile."""

    segs = TileSEGSBuilder().build(
        _image(4, 8),
        _controls(bbox_size=4, mask_irregularity=0.5, irregular_mask_mode="Reuse fast"),
    )

    assert torch.equal(
        torch.as_tensor(segs[1][0].cropped_mask),
        torch.as_tensor(segs[1][1].cropped_mask),
    )


def test_irregular_all_random_mode_varies_mask_pattern() -> None:
    """All-random modes generate a distinct mask pattern for each tile."""

    segs = TileSEGSBuilder().build(
        _image(4, 8),
        _controls(
            bbox_size=4,
            mask_irregularity=0.5,
            irregular_mask_mode="All random fast",
        ),
    )

    assert not torch.equal(
        torch.as_tensor(segs[1][0].cropped_mask),
        torch.as_tensor(segs[1][1].cropped_mask),
    )


def _image(height: int, width: int) -> torch.Tensor:
    """Return a black BHWC image."""

    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def _controls(
    *,
    bbox_size: int,
    crop_factor: float = 1.0,
    min_overlap: int = 0,
    filter_segs_dilation: int = 0,
    mask_irregularity: float = 0.0,
    irregular_mask_mode: str = "Reuse fast",
) -> TileSEGSControls:
    """Return valid tile controls for tests."""

    return TileSEGSControls(
        bbox_size=bbox_size,
        crop_factor=crop_factor,
        min_overlap=min_overlap,
        filter_segs_dilation=filter_segs_dilation,
        mask_irregularity=mask_irregularity,
        irregular_mask_mode=irregular_mask_mode,
    )
