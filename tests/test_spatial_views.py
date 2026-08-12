"""Verify immutable spatial view and view-major layout invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)


def test_spatial_view_records_source_and_model_geometry_immutably() -> None:
    """Expose exact rectangle edges while preventing mutation."""

    view = _tile(x=4, y=8, width=16, height=12, model_width=8, model_height=6)

    assert view.source_right == 20
    assert view.source_bottom == 20
    with pytest.raises(FrozenInstanceError):
        view.source_x = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        {"source_x": -1},
        {"source_y": -1},
        {"source_width": 0},
        {"source_height": 0},
        {"model_width": 0},
        {"model_height": 0},
    ],
)
def test_spatial_view_rejects_invalid_geometry(values: dict[str, int]) -> None:
    """Reject negative origins and non-positive source or model dimensions."""

    with pytest.raises(ValueError):
        _tile(**values)


def test_spatial_view_rejects_untyped_kind() -> None:
    """Require callers to select one explicit view-kind value."""

    with pytest.raises(TypeError, match="SpatialViewKind"):
        SpatialView(
            kind="tile",  # type: ignore[arg-type]
            source_x=0,
            source_y=0,
            source_width=8,
            source_height=8,
            model_width=8,
            model_height=8,
        )


def test_full_view_preserves_source_dimensions() -> None:
    """Reject resized full views while allowing reduced Contextual global views."""

    with pytest.raises(ValueError, match="preserve its source dimensions"):
        SpatialView(
            kind=SpatialViewKind.FULL,
            source_x=0,
            source_y=0,
            source_width=16,
            source_height=12,
            model_width=8,
            model_height=6,
        )

    contextual_global = SpatialView(
        kind=SpatialViewKind.CONTEXTUAL_GLOBAL,
        source_x=0,
        source_y=0,
        source_width=16,
        source_height=12,
        model_width=8,
        model_height=6,
    )
    assert contextual_global.model_width == 8
    assert contextual_global.model_height == 6


def test_layout_expands_views_in_exact_view_major_order() -> None:
    """Keep each view's complete source batch contiguous in model-call order."""

    left = _tile(x=0, width=8)
    right = _tile(x=8, width=8)
    layout = SpatialBatchLayout(
        canvas_width=16,
        canvas_height=8,
        views=(left, right),
        input_batch_size=3,
    )

    assert layout.view_count == 2
    assert layout.expanded_batch_size == 6
    assert layout.expanded_views == (left, left, left, right, right, right)
    assert layout.expanded_view_indices == (0, 0, 0, 1, 1, 1)
    assert layout.expanded_source_batch_indices == (0, 1, 2, 0, 1, 2)
    assert layout.expanded_index(0, 2) == 2
    assert layout.expanded_index(1, 0) == 3
    with pytest.raises(FrozenInstanceError):
        layout.input_batch_size = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("canvas_width", 0, "canvas dimensions"),
        ("canvas_height", 0, "canvas dimensions"),
        ("input_batch_size", 0, "input batch size"),
        ("views", (), "at least one view"),
    ],
)
def test_layout_rejects_invalid_dimensions_or_empty_views(
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject layouts that cannot describe a non-empty model call."""

    values: dict[str, object] = {
        "canvas_width": 8,
        "canvas_height": 8,
        "views": (_tile(),),
        "input_batch_size": 1,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        SpatialBatchLayout(**values)  # type: ignore[arg-type]


def test_layout_requires_an_immutable_typed_view_tuple() -> None:
    """Reject mutable containers and values that are not spatial views."""

    with pytest.raises(TypeError, match="immutable tuple"):
        SpatialBatchLayout(8, 8, [_tile()], 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SpatialView values"):
        SpatialBatchLayout(8, 8, (object(),), 1)  # type: ignore[arg-type]


def test_layout_rejects_out_of_canvas_or_mixed_views() -> None:
    """Reject source overflow and heterogeneous model-call view semantics."""

    with pytest.raises(ValueError, match="inside the canvas"):
        SpatialBatchLayout(8, 8, (_tile(x=4, width=8),), 1)

    full = SpatialView(
        SpatialViewKind.FULL,
        0,
        0,
        8,
        8,
        8,
        8,
    )
    with pytest.raises(ValueError, match="cannot mix view kinds"):
        SpatialBatchLayout(8, 8, (full, _tile()), 1)


@pytest.mark.parametrize(
    "kind",
    [SpatialViewKind.FULL, SpatialViewKind.CONTEXTUAL_GLOBAL],
)
def test_full_source_layout_requires_one_complete_canvas_view(
    kind: SpatialViewKind,
) -> None:
    """Require full and Contextual-global calls to describe the whole canvas."""

    incomplete = SpatialView(kind, 1, 0, 7, 8, 7, 8)
    with pytest.raises(ValueError, match="complete canvas"):
        SpatialBatchLayout(8, 8, (incomplete,), 1)

    complete = SpatialView(kind, 0, 0, 8, 8, 8, 8)
    with pytest.raises(ValueError, match="requires one view"):
        SpatialBatchLayout(8, 8, (complete, complete), 1)


def test_layout_rejects_expanded_indices_outside_either_axis() -> None:
    """Fail explicitly for invalid view or source-batch indices."""

    layout = SpatialBatchLayout(8, 8, (_tile(),), 2)

    with pytest.raises(IndexError, match="view index"):
        layout.expanded_index(1, 0)
    with pytest.raises(IndexError, match="batch index"):
        layout.expanded_index(0, 2)


def _tile(
    *,
    x: int = 0,
    y: int = 0,
    width: int = 8,
    height: int = 8,
    model_width: int = 8,
    model_height: int = 8,
    source_x: int | None = None,
    source_y: int | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
) -> SpatialView:
    """Return one tile view with selectively overridden source fields."""

    return SpatialView(
        kind=SpatialViewKind.TILE,
        source_x=x if source_x is None else source_x,
        source_y=y if source_y is None else source_y,
        source_width=width if source_width is None else source_width,
        source_height=height if source_height is None else source_height,
        model_width=model_width,
        model_height=model_height,
    )
