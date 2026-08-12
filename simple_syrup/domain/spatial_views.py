# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable spatial model views and view-major batch layouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SpatialViewKind(StrEnum):
    """Identify how one model view relates to the canonical latent canvas."""

    FULL = "full"
    TILE = "tile"
    CONTEXTUAL_GLOBAL = "contextual_global"


@dataclass(frozen=True, slots=True)
class SpatialView:
    """Describe one source rectangle evaluated at one model spatial shape."""

    kind: SpatialViewKind
    source_x: int
    source_y: int
    source_width: int
    source_height: int
    model_width: int
    model_height: int

    def __post_init__(self) -> None:
        """Reject invalid or internally inconsistent view geometry."""

        if not isinstance(self.kind, SpatialViewKind):
            raise TypeError("Spatial view kind must be a SpatialViewKind value.")
        if self.source_x < 0 or self.source_y < 0:
            raise ValueError("Spatial view source coordinates must be non-negative.")
        if self.source_width < 1 or self.source_height < 1:
            raise ValueError("Spatial view source dimensions must be positive.")
        if self.model_width < 1 or self.model_height < 1:
            raise ValueError("Spatial view model dimensions must be positive.")
        if self.kind is SpatialViewKind.FULL and (
            self.source_width != self.model_width
            or self.source_height != self.model_height
        ):
            raise ValueError("A full spatial view must preserve its source dimensions.")

    @property
    def source_right(self) -> int:
        """Return the exclusive source rectangle right edge."""

        return self.source_x + self.source_width

    @property
    def source_bottom(self) -> int:
        """Return the exclusive source rectangle bottom edge."""

        return self.source_y + self.source_height


@dataclass(frozen=True, slots=True)
class SpatialBatchLayout:
    """Describe ordered spatial views expanded over one source model batch."""

    canvas_width: int
    canvas_height: int
    views: tuple[SpatialView, ...]
    input_batch_size: int

    def __post_init__(self) -> None:
        """Validate canvas containment and homogeneous model-call semantics."""

        if self.canvas_width < 1 or self.canvas_height < 1:
            raise ValueError("Spatial layout canvas dimensions must be positive.")
        if self.input_batch_size < 1:
            raise ValueError("Spatial layout input batch size must be positive.")
        if not isinstance(self.views, tuple):
            raise TypeError("Spatial layout views must be an immutable tuple.")
        if not self.views:
            raise ValueError("Spatial layout requires at least one view.")
        if not all(isinstance(view, SpatialView) for view in self.views):
            raise TypeError("Spatial layout views must contain SpatialView values.")

        view_kind = self.views[0].kind
        if any(view.kind is not view_kind for view in self.views):
            raise ValueError("One spatial model call cannot mix view kinds.")
        for view in self.views:
            if (
                view.source_right > self.canvas_width
                or view.source_bottom > self.canvas_height
            ):
                raise ValueError(
                    "Spatial view source rectangle must remain inside the canvas."
                )

        if view_kind in {
            SpatialViewKind.FULL,
            SpatialViewKind.CONTEXTUAL_GLOBAL,
        }:
            if len(self.views) != 1:
                raise ValueError("A full-source spatial layout requires one view.")
            view = self.views[0]
            if (
                view.source_x != 0
                or view.source_y != 0
                or view.source_width != self.canvas_width
                or view.source_height != self.canvas_height
            ):
                raise ValueError(
                    "A full-source spatial view must cover the complete canvas."
                )

    @property
    def view_count(self) -> int:
        """Return the number of ordered spatial views."""

        return len(self.views)

    @property
    def expanded_batch_size(self) -> int:
        """Return the model batch size after view-major expansion."""

        return self.view_count * self.input_batch_size

    @property
    def expanded_views(self) -> tuple[SpatialView, ...]:
        """Repeat each view for its contiguous source-batch group."""

        return tuple(
            view
            for view in self.views
            for _source_batch_index in range(self.input_batch_size)
        )

    @property
    def expanded_view_indices(self) -> tuple[int, ...]:
        """Return the view index for every expanded model-batch entry."""

        return tuple(
            view_index
            for view_index in range(self.view_count)
            for _source_batch_index in range(self.input_batch_size)
        )

    @property
    def expanded_source_batch_indices(self) -> tuple[int, ...]:
        """Return the source-batch index for every expanded model-batch entry."""

        return tuple(
            source_batch_index
            for _view in self.views
            for source_batch_index in range(self.input_batch_size)
        )

    def expanded_index(self, view_index: int, source_batch_index: int) -> int:
        """Return one view-major model-batch index after validating both axes."""

        if not 0 <= view_index < self.view_count:
            raise IndexError("Spatial view index is outside the layout.")
        if not 0 <= source_batch_index < self.input_batch_size:
            raise IndexError("Source batch index is outside the layout.")
        return view_index * self.input_batch_size + source_batch_index
