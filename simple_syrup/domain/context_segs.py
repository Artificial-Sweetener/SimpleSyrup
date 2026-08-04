# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project evaluated latent context windows into lazily materialized SEGS."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from threading import Lock
from typing import TypeAlias, overload

import torch

from .segs import BoundingBox, CropRegion, Segment, SegsHeader
from .tiled_diffusion import TiledDiffusionPlan

ContextSegs: TypeAlias = tuple[SegsHeader, "ContextSegmentSequence"]


class ContextSegmentSequence(Sequence[Segment]):
    """Delay large rectangular mask allocation until a SEGS consumer reads it."""

    def __init__(self, windows: Iterable[CropRegion]) -> None:
        """Store deterministic context windows without allocating their masks."""

        self._windows = tuple(windows)
        self._materialized: tuple[Segment, ...] | None = None
        self._lock = Lock()

    @property
    def windows(self) -> tuple[CropRegion, ...]:
        """Return immutable projected windows without forcing mask allocation."""

        return self._windows

    @property
    def is_materialized(self) -> bool:
        """Report whether a downstream consumer has requested concrete segments."""

        return self._materialized is not None

    def __len__(self) -> int:
        """Return the number of contexts without materializing masks."""

        return len(self._windows)

    @overload
    def __getitem__(self, index: int) -> Segment: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Segment, ...]: ...

    def __getitem__(self, index: int | slice) -> Segment | tuple[Segment, ...]:
        """Materialize masks and return one context or a context slice."""

        return self._segments()[index]

    def __iter__(self) -> Iterator[Segment]:
        """Materialize masks once and iterate contexts in evaluation order."""

        return iter(self._segments())

    def _segments(self) -> tuple[Segment, ...]:
        """Create full rectangular masks once on first downstream access."""

        if self._materialized is not None:
            return self._materialized
        with self._lock:
            if self._materialized is None:
                self._materialized = tuple(
                    _segment_from_window(window, index)
                    for index, window in enumerate(self._windows, start=1)
                )
        return self._materialized


def context_segs_from_tile_plan(
    plan: TiledDiffusionPlan,
    *,
    image_height: int,
    image_width: int,
) -> ContextSegs:
    """Return lazy rectangular SEGS for every non-global evaluated tile."""

    _validate_image_dimensions(image_height, image_width)
    windows = tuple(
        _project_tile(
            x=tile.x,
            y=tile.y,
            width=tile.width,
            height=tile.height,
            latent_width=plan.latent_width,
            latent_height=plan.latent_height,
            image_width=image_width,
            image_height=image_height,
        )
        for tile in plan.tiles
    )
    return (image_height, image_width), ContextSegmentSequence(windows)


def merge_context_segs(values: Iterable[ContextSegs]) -> ContextSegs:
    """Combine batched context windows without duplicates or mask allocation."""

    items = tuple(values)
    if not items:
        raise ValueError("Context SEGS requires at least one context plan.")
    header = items[0][0]
    if any(item[0] != header for item in items[1:]):
        raise ValueError(
            "Context SEGS requires batched image dimensions to match exactly."
        )
    windows = list(items[0][1].windows)
    seen = set(windows)
    for _header, segments in items[1:]:
        for window in segments.windows:
            if window not in seen:
                windows.append(window)
                seen.add(window)
    return header, ContextSegmentSequence(windows)


def _project_tile(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    latent_width: int,
    latent_height: int,
    image_width: int,
    image_height: int,
) -> CropRegion:
    """Project one latent rectangle outward into integer image coordinates."""

    left = math.floor(x * image_width / latent_width)
    top = math.floor(y * image_height / latent_height)
    right = math.ceil((x + width) * image_width / latent_width)
    bottom = math.ceil((y + height) * image_height / latent_height)
    return CropRegion(
        max(0, min(image_width - 1, left)),
        max(0, min(image_height - 1, top)),
        max(1, min(image_width, right)),
        max(1, min(image_height, bottom)),
    )


def _segment_from_window(window: CropRegion, index: int) -> Segment:
    """Materialize one Impact-compatible full rectangular context mask."""

    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones(
            (window.height, window.width),
            dtype=torch.uint8,
        ),
        confidence=1.0,
        crop_region=window,
        bbox=BoundingBox(*window),
        label=f"context_{index:03d}",
    )


def _validate_image_dimensions(height: int, width: int) -> None:
    """Reject image geometry that cannot host projected context rectangles."""

    if height < 1 or width < 1:
        raise ValueError("Context SEGS image dimensions must be positive.")
