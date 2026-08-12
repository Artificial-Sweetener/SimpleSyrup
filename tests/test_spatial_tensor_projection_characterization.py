"""Characterize the focused final-axis spatial tensor projection owner."""

from __future__ import annotations

import torch

from simple_syrup.domain.spatial_views import SpatialView, SpatialViewKind
from simple_syrup.domain.tiled_diffusion import LatentTile
from simple_syrup.runtime.spatial_tensor_projection import (
    resize_spatial_tensor,
    spatial_tile_slicer,
    spatial_view_slicer,
)


def test_tile_slicer_crops_final_axes_for_bchw_and_bcdhw() -> None:
    """Preserve every leading axis while slicing one planned tile rectangle."""

    tile = LatentTile(x=2, y=1, width=3, height=2)
    bchw = torch.arange(1 * 2 * 4 * 6).reshape((1, 2, 4, 6))
    bcdhw = bchw.unsqueeze(2)

    assert torch.equal(
        bchw[spatial_tile_slicer(tile, bchw.ndim)],
        bchw[:, :, 1:3, 2:5],
    )
    assert torch.equal(
        bcdhw[spatial_tile_slicer(tile, bcdhw.ndim)],
        bcdhw[:, :, :, 1:3, 2:5],
    )


def test_view_slicer_crops_source_rectangle_not_model_shape() -> None:
    """Use source bounds for cropping before a view is resized for its model."""

    view = SpatialView(
        kind=SpatialViewKind.TILE,
        source_x=1,
        source_y=2,
        source_width=4,
        source_height=2,
        model_width=2,
        model_height=4,
    )
    tensor = torch.arange(1 * 2 * 1 * 5 * 7).reshape((1, 2, 1, 5, 7))

    cropped = tensor[spatial_view_slicer(view, tensor.ndim)]

    assert torch.equal(cropped, tensor[:, :, :, 2:4, 1:5])
    assert cropped.shape[-2:] == (view.source_height, view.source_width)
    assert cropped.shape[-2:] != (view.model_height, view.model_width)


def test_resize_returns_identity_when_spatial_shape_already_matches() -> None:
    """Avoid allocation or interpolation for an already matching tensor."""

    tensor = torch.ones((2, 3, 1, 4, 6), dtype=torch.float16)

    resized = resize_spatial_tensor(tensor, height=4, width=6, mode="bilinear")

    assert resized is tensor


def test_resize_changes_only_final_axes_for_bchw() -> None:
    """Preserve batch/channel values while resizing BCHW height and width."""

    tensor = torch.tensor([[[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]]])

    resized = resize_spatial_tensor(tensor, height=1, width=2, mode="nearest-exact")

    assert resized.shape == (1, 1, 1, 2)
    assert torch.equal(resized, torch.tensor([[[[1.0, 2.0]]]]))


def test_resize_preserves_all_leading_bcdhw_axes_and_dtype() -> None:
    """Flatten only for interpolation and restore BCDHW leading dimensions."""

    tensor = torch.stack(
        (
            torch.ones((3, 1, 2, 2), dtype=torch.float16),
            torch.full((3, 1, 2, 2), 2.0, dtype=torch.float16),
        )
    )

    resized = resize_spatial_tensor(tensor, height=4, width=6, mode="bilinear")

    assert resized.shape == (2, 3, 1, 4, 6)
    assert resized.dtype == torch.float16
    assert torch.equal(resized[0], torch.ones_like(resized[0]))
    assert torch.equal(resized[1], torch.full_like(resized[1], 2.0))
