# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify canonical regional masks project by explicit spatial policy."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
import torch.nn.functional as functional
from _pytest.monkeypatch import MonkeyPatch

from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
    RegionalMaskProjector,
)


def test_view_projection_crops_before_continuous_coverage_downsampling() -> None:
    """Average only the selected source rectangle into the view model grid."""

    bank = _bank(torch.arange(24, dtype=torch.float32).reshape((1, 4, 6)) / 23.0)
    layout = _layout(
        SpatialView(SpatialViewKind.TILE, 2, 0, 4, 4, 2, 2),
        canvas_width=6,
        canvas_height=4,
    )

    projected = RegionalMaskProjector().project_view(
        bank=bank,
        layout=layout,
        view_index=0,
        form=RegionalMaskForm.PLANNING,
        mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
    )

    expected = functional.interpolate(
        bank.planning_masks[:, :, 2:].unsqueeze(1),
        size=(2, 2),
        mode="area",
    ).squeeze(1)
    torch.testing.assert_close(projected, expected)


def test_query_projection_selects_conditioning_form() -> None:
    """Select feathered conditioning masks independently from planning masks."""

    planning = torch.zeros((1, 2, 2))
    conditioning = torch.ones((1, 2, 2))
    bank = _bank(planning, conditioning=conditioning)
    layout = _layout(
        SpatialView(SpatialViewKind.FULL, 0, 0, 2, 2, 2, 2),
        canvas_width=2,
        canvas_height=2,
    )

    projected = RegionalMaskProjector().project_query_grid(
        bank=bank,
        layout=layout,
        view_index=0,
        query_height=2,
        query_width=2,
        form=RegionalMaskForm.CONDITIONING,
        mode=RegionalMaskProjectionMode.SOFT,
    )

    assert torch.equal(projected, conditioning)


def test_projection_selects_one_ordered_view_from_a_tile_batch() -> None:
    """Address the requested view without mixing neighboring tile geometry."""

    masks = torch.arange(8, dtype=torch.float32).reshape((1, 2, 4)) / 7.0
    bank = _bank(masks)
    layout = SpatialBatchLayout(
        canvas_width=4,
        canvas_height=2,
        views=(
            SpatialView(SpatialViewKind.TILE, 0, 0, 2, 2, 2, 2),
            SpatialView(SpatialViewKind.TILE, 2, 0, 2, 2, 2, 2),
        ),
        input_batch_size=2,
    )

    projected = RegionalMaskProjector().project_view(
        bank=bank,
        layout=layout,
        view_index=1,
        form=RegionalMaskForm.PLANNING,
        mode=RegionalMaskProjectionMode.HARD_PRESERVING,
    )

    assert torch.equal(projected, masks[:, :, 2:])


def test_reduced_global_projection_uses_full_canvas_coverage() -> None:
    """Average the complete canvas into a Contextual global model grid."""

    masks = torch.arange(16, dtype=torch.float32).reshape((1, 4, 4)) / 15.0
    bank = _bank(masks)
    layout = _layout(
        SpatialView(SpatialViewKind.CONTEXTUAL_GLOBAL, 0, 0, 4, 4, 2, 2),
        canvas_width=4,
        canvas_height=4,
    )

    projected = RegionalMaskProjector().project_view(
        bank=bank,
        layout=layout,
        view_index=0,
        form=RegionalMaskForm.PLANNING,
        mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
    )

    expected = functional.interpolate(
        masks.unsqueeze(1), size=(2, 2), mode="area"
    ).squeeze(1)
    torch.testing.assert_close(projected, expected)


@pytest.mark.parametrize(
    ("mode", "torch_mode"),
    [
        (RegionalMaskProjectionMode.SOFT, "bilinear"),
        (RegionalMaskProjectionMode.HARD_PRESERVING, "nearest-exact"),
        (RegionalMaskProjectionMode.NEAREST, "nearest"),
    ],
)
def test_query_projection_uses_requested_soft_or_hard_interpolation(
    mode: RegionalMaskProjectionMode,
    torch_mode: str,
) -> None:
    """Keep soft and explicitly hard-preserving resampling paths distinct."""

    masks = torch.tensor([[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]])
    bank = _bank(masks)
    layout = _layout(
        SpatialView(SpatialViewKind.FULL, 0, 0, 3, 2, 3, 2),
        canvas_width=3,
        canvas_height=2,
    )

    projected = RegionalMaskProjector().project_query_grid(
        bank=bank,
        layout=layout,
        view_index=0,
        query_height=4,
        query_width=5,
        form=RegionalMaskForm.PLANNING,
        mode=mode,
    )
    interpolation_args: dict[str, Any] = {}
    if torch_mode == "bilinear":
        interpolation_args["align_corners"] = False
    expected = functional.interpolate(
        masks.unsqueeze(1),
        size=(4, 5),
        mode=torch_mode,
        **interpolation_args,
    ).squeeze(1)

    torch.testing.assert_close(projected, expected)


def test_mixed_continuous_projection_uses_area_then_bilinear() -> None:
    """Preserve coverage on a shrinking axis before growing the other axis."""

    masks = torch.arange(8, dtype=torch.float32).reshape((1, 4, 2)) / 7.0
    bank = _bank(masks)
    layout = _layout(
        SpatialView(SpatialViewKind.FULL, 0, 0, 2, 4, 2, 4),
        canvas_width=2,
        canvas_height=4,
    )

    projected = RegionalMaskProjector().project_query_grid(
        bank=bank,
        layout=layout,
        view_index=0,
        query_height=2,
        query_width=4,
        form=RegionalMaskForm.PLANNING,
        mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
    )

    downsampled = functional.interpolate(masks.unsqueeze(1), size=(2, 2), mode="area")
    expected = functional.interpolate(
        downsampled,
        size=(2, 4),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    torch.testing.assert_close(projected, expected)


@pytest.mark.parametrize(
    ("form", "mode", "message"),
    [
        (
            cast(Any, "planning"),
            RegionalMaskProjectionMode.SOFT,
            "form must be a RegionalMaskForm",
        ),
        (
            RegionalMaskForm.PLANNING,
            cast(Any, "soft"),
            "mode must be a RegionalMaskProjectionMode",
        ),
    ],
)
def test_projection_rejects_untyped_policy_selectors(
    form: RegionalMaskForm,
    mode: RegionalMaskProjectionMode,
    message: str,
) -> None:
    """Require explicit typed form and interpolation policy at every callsite."""

    bank = _bank(torch.ones((1, 2, 2)))
    layout = _layout(
        SpatialView(SpatialViewKind.FULL, 0, 0, 2, 2, 2, 2),
        canvas_width=2,
        canvas_height=2,
    )

    with pytest.raises(TypeError, match=message):
        RegionalMaskProjector().project_view(
            bank=bank,
            layout=layout,
            view_index=0,
            form=form,
            mode=mode,
        )


def test_projection_rejects_canvas_mismatch() -> None:
    """Require the mask bank and spatial layout to share one canvas authority."""

    bank = _bank(torch.ones((1, 2, 2)))
    layout = _layout(
        SpatialView(SpatialViewKind.TILE, 0, 0, 2, 2, 2, 2),
        canvas_width=3,
        canvas_height=2,
    )

    with pytest.raises(ValueError, match="canvas must match"):
        RegionalMaskProjector().project_view(
            bank=bank,
            layout=layout,
            view_index=0,
            form=RegionalMaskForm.PLANNING,
            mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
        )


@pytest.mark.parametrize(
    ("view_index", "height", "width", "error", "message"),
    [
        (-1, 2, 2, IndexError, "view index is outside"),
        (1, 2, 2, IndexError, "view index is outside"),
        (0, 0, 2, ValueError, "query-grid dimensions must be positive"),
        (0, 2, 0, ValueError, "query-grid dimensions must be positive"),
    ],
)
def test_query_projection_rejects_invalid_addressing(
    view_index: int,
    height: int,
    width: int,
    error: type[Exception],
    message: str,
) -> None:
    """Fail before interpolation for invalid view or query-grid coordinates."""

    bank = _bank(torch.ones((1, 2, 2)))
    layout = _layout(
        SpatialView(SpatialViewKind.FULL, 0, 0, 2, 2, 2, 2),
        canvas_width=2,
        canvas_height=2,
    )

    with pytest.raises(error, match=message):
        RegionalMaskProjector().project_query_grid(
            bank=bank,
            layout=layout,
            view_index=view_index,
            query_height=height,
            query_width=width,
            form=RegionalMaskForm.PLANNING,
            mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
        )


def test_projection_rejects_non_finite_interpolation_result(
    monkeypatch: MonkeyPatch,
) -> None:
    """Fail closed when the tensor interpolation boundary returns invalid data."""

    bank = _bank(torch.ones((1, 2, 2)))
    layout = _layout(
        SpatialView(SpatialViewKind.FULL, 0, 0, 2, 2, 2, 2),
        canvas_width=2,
        canvas_height=2,
    )

    def invalid_interpolation(*_args: object, **_kwargs: object) -> torch.Tensor:
        """Return one invalid projected mask from the mocked tensor boundary."""

        return torch.full((1, 1, 3, 3), float("nan"))

    monkeypatch.setattr(
        "simple_syrup.masking.regional_mask_projection.functional.interpolate",
        invalid_interpolation,
    )
    with pytest.raises(ValueError, match="contain non-finite values"):
        RegionalMaskProjector().project_query_grid(
            bank=bank,
            layout=layout,
            view_index=0,
            query_height=3,
            query_width=3,
            form=RegionalMaskForm.PLANNING,
            mode=RegionalMaskProjectionMode.SOFT,
        )


def _bank(
    planning: torch.Tensor,
    *,
    conditioning: torch.Tensor | None = None,
) -> RegionalMaskBank:
    """Return a valid bank matching the supplied planning-mask canvas."""

    conditioning_masks = (
        planning.to(device=planning.device, dtype=planning.dtype, copy=True)
        if conditioning is None
        else conditioning
    )
    return RegionalMaskBank(
        planning_masks=planning,
        conditioning_masks=conditioning_masks,
        canvas_width=int(planning.shape[-1]),
        canvas_height=int(planning.shape[-2]),
    )


def _layout(
    view: SpatialView,
    *,
    canvas_width: int,
    canvas_height: int,
) -> SpatialBatchLayout:
    """Return a one-view canonical layout for projection tests."""

    return SpatialBatchLayout(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        views=(view,),
        input_batch_size=1,
    )
