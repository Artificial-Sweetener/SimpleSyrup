"""Verify exact SpatialBatchLayout construction from tiled plans."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from simple_syrup.domain.regional_tiled_diffusion import (
    build_region_constrained_tiled_diffusion_plan,
)
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.domain.segs_tiled_diffusion import (
    build_segs_guided_tiled_diffusion_plan,
)
from simple_syrup.domain.spatial_views import SpatialViewKind
from simple_syrup.domain.tiled_diffusion import (
    TiledDiffusionPlan,
    build_tiled_diffusion_plan,
)
from simple_syrup.runtime.spatial_model_arguments import tiled_batch_layout

PlanFactory = Callable[[], TiledDiffusionPlan]


@pytest.mark.parametrize(
    "plan_factory",
    [
        lambda: build_tiled_diffusion_plan(12, 8, 8, 6, 2, 2),
        lambda: build_segs_guided_tiled_diffusion_plan(
            segs=((8, 12), (_full_segment(8, 12),)),
            latent_width=12,
            latent_height=8,
            tile_width=8,
            tile_height=6,
            overlap=2,
            tile_batch_size=2,
        ),
        lambda: build_region_constrained_tiled_diffusion_plan(
            region_masks=_left_right_masks(8, 12),
            segs=None,
            latent_width=12,
            latent_height=8,
            tile_width=8,
            tile_height=6,
            overlap=2,
            tile_batch_size=2,
        ),
    ],
    ids=("regular-edge", "segs-guided", "region-constrained"),
)
def test_tiled_layout_preserves_every_authoritative_tile_in_order(
    plan_factory: PlanFactory,
) -> None:
    """Map regular and semantic tile batches to exact ordered TILE views."""

    plan = plan_factory()
    assert plan.batches

    for batch in plan.batches:
        layout = tiled_batch_layout(
            tiles=batch,
            input_batch_size=3,
            latent_height=plan.latent_height,
            latent_width=plan.latent_width,
        )

        assert layout.canvas_width == plan.latent_width
        assert layout.canvas_height == plan.latent_height
        assert layout.input_batch_size == 3
        assert layout.view_count == len(batch)
        assert [
            (
                view.kind,
                view.source_x,
                view.source_y,
                view.source_width,
                view.source_height,
                view.model_width,
                view.model_height,
            )
            for view in layout.views
        ] == [
            (
                SpatialViewKind.TILE,
                tile.x,
                tile.y,
                tile.width,
                tile.height,
                tile.width,
                tile.height,
            )
            for tile in batch
        ]
        assert layout.expanded_view_indices == tuple(
            view_index for view_index in range(len(batch)) for _source_index in range(3)
        )
        assert layout.expanded_source_batch_indices == tuple(range(3)) * len(batch)


@pytest.mark.parametrize("tile_batch_size", [1, 2, 4, 8])
def test_tiled_layout_keeps_view_major_order_for_supported_batch_sizes(
    tile_batch_size: int,
) -> None:
    """Prove every supported tile batch expands over a multi-item latent batch."""

    plan = build_tiled_diffusion_plan(
        latent_width=64,
        latent_height=32,
        tile_width=16,
        tile_height=16,
        overlap=0,
        tile_batch_size=tile_batch_size,
    )
    assert max(len(batch) for batch in plan.batches) == tile_batch_size

    for batch in plan.batches:
        layout = tiled_batch_layout(
            tiles=batch,
            input_batch_size=2,
            latent_height=plan.latent_height,
            latent_width=plan.latent_width,
        )

        assert layout.expanded_view_indices == tuple(
            view_index for view_index in range(len(batch)) for _ in range(2)
        )
        assert layout.expanded_source_batch_indices == tuple(range(2)) * len(batch)
        assert layout.expanded_batch_size == len(batch) * 2


def _full_segment(height: int, width: int) -> Segment:
    """Return one full-canvas segment."""

    region = CropRegion(0, 0, width, height)
    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones((height, width)),
        confidence=1.0,
        crop_region=region,
        bbox=BoundingBox(0, 0, width, height),
        label="region",
    )


def _left_right_masks(height: int, width: int) -> torch.Tensor:
    """Return two masks that partition the canvas horizontally."""

    masks = torch.zeros((2, height, width))
    masks[0, :, : width // 2] = 1.0
    masks[1, :, width // 2 :] = 1.0
    return masks
