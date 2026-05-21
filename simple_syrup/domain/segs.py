# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Domain model and adapters for Impact-compatible SEGS values."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol, TypeAlias, cast


class CropRegion(NamedTuple):
    """Represent a crop region as left, top, right, bottom coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Return the region width."""

        return self.right - self.left

    @property
    def height(self) -> int:
        """Return the region height."""

        return self.bottom - self.top


class BoundingBox(NamedTuple):
    """Represent a detection box as left, top, right, bottom coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Return the box width."""

        return self.right - self.left

    @property
    def height(self) -> int:
        """Return the box height."""

        return self.bottom - self.top


@dataclass(frozen=True)
class Segment:
    """Represent one detected region in an Impact-compatible shape."""

    cropped_image: object | None
    cropped_mask: object
    confidence: float
    crop_region: CropRegion
    bbox: BoundingBox
    label: str
    control_net_wrapper: object | None = None


class SegmentLike(Protocol):
    """Describe the attribute surface required for SEGS compatibility."""

    cropped_image: object | None
    cropped_mask: object
    confidence: float
    crop_region: object
    bbox: object
    label: str
    control_net_wrapper: object | None


SegsHeader: TypeAlias = tuple[int, int]
NativeSegs: TypeAlias = tuple[SegsHeader, tuple[Segment, ...]]
ImpactSegs: TypeAlias = tuple[SegsHeader, list[Segment]]
NativeSegsGroup: TypeAlias = tuple[NativeSegs, ...]
SortKey: TypeAlias = tuple[float, float, int, int, int]

SORT_ORDER_OPTIONS: tuple[str, ...] = (
    "largest to smallest",
    "smallest to largest",
    "widest to thinnest",
    "thinnest to widest",
    "tallest to shortest",
    "shortest to tallest",
    "top to bottom",
    "bottom to top",
    "left to right",
    "right to left",
    "highest confidence first",
    "lowest confidence first",
)


def coerce_segs(value: object) -> NativeSegs:
    """Convert a native or Impact-style SEGS value into native immutable SEGS."""

    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("SEGS must be a tuple of (header, segments).")

    header = _coerce_header(value[0])
    raw_segments = value[1]
    if not isinstance(raw_segments, Iterable):
        raise ValueError("SEGS segments must be iterable.")

    return header, tuple(coerce_segment(segment) for segment in raw_segments)


def coerce_segment(value: object) -> Segment:
    """Convert a native or attribute-compatible segment into `Segment`."""

    if isinstance(value, Segment):
        _validate_region(value.crop_region, "crop_region")
        _validate_box(value.bbox, "bbox")
        return value

    required = (
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    )
    missing = [name for name in required if not hasattr(value, name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"SEG item is missing required attribute(s): {names}.")

    segment_like = cast(SegmentLike, value)
    return Segment(
        cropped_image=segment_like.cropped_image,
        cropped_mask=segment_like.cropped_mask,
        confidence=float(segment_like.confidence),
        crop_region=_coerce_region(segment_like.crop_region, "crop_region"),
        bbox=_coerce_box(segment_like.bbox, "bbox"),
        label=str(segment_like.label),
        control_net_wrapper=segment_like.control_net_wrapper,
    )


def to_impact_compatible_segs(segs: NativeSegs) -> ImpactSegs:
    """Return raw tuple/list SEGS that Impact-style consumers can read."""

    header, segments = coerce_segs(segs)
    return header, list(segments)


def coerce_segs_group(value: object) -> NativeSegsGroup:
    """Convert a single SEGS or a Comfy list of SEGS into native SEGS."""

    if _looks_like_segs(value):
        return (coerce_segs(value),)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("SEGS group must be a SEGS payload or a sequence of SEGS.")
    if len(value) == 0:
        raise ValueError("SEGS group must contain one or more SEGS payloads.")

    group: list[NativeSegs] = []
    for index, item in enumerate(value, start=1):
        try:
            group.append(coerce_segs(item))
        except ValueError as exc:
            raise ValueError(f"SEGS group item {index} is invalid: {exc}") from exc
    return tuple(group)


def to_impact_compatible_segs_group(segs_group: NativeSegsGroup) -> list[ImpactSegs]:
    """Return a list of Impact-compatible SEGS payloads."""

    return [to_impact_compatible_segs(segs) for segs in segs_group]


def sort_segs(segs: NativeSegs, sort_order: str) -> NativeSegs:
    """Return native SEGS sorted by a plain-English policy."""

    if sort_order not in SORT_ORDER_OPTIONS:
        raise ValueError(f"Unknown SEGS sort order: '{sort_order}'.")
    header, segments = coerce_segs(segs)
    indexed_segments = tuple(enumerate(segments))
    sorted_segments = sorted(
        indexed_segments,
        key=lambda item: _sort_key(item[1], item[0], sort_order),
    )
    return header, tuple(segment for _index, segment in sorted_segments)


def _sort_key(segment: Segment, index: int, sort_order: str) -> SortKey:
    """Build a deterministic sort key for one segment."""

    region = segment.crop_region
    confidence_desc = -float(segment.confidence)
    if sort_order == "highest confidence first":
        return confidence_desc, float(region.top), region.left, region.left, index
    if sort_order == "lowest confidence first":
        return (
            float(segment.confidence),
            float(region.top),
            region.left,
            region.left,
            index,
        )

    if sort_order == "largest to smallest":
        primary = -float(region.width * region.height)
    elif sort_order == "smallest to largest":
        primary = float(region.width * region.height)
    elif sort_order == "widest to thinnest":
        primary = -float(region.width)
    elif sort_order == "thinnest to widest":
        primary = float(region.width)
    elif sort_order == "tallest to shortest":
        primary = -float(region.height)
    elif sort_order == "shortest to tallest":
        primary = float(region.height)
    elif sort_order == "top to bottom":
        primary = float(region.top)
    elif sort_order == "bottom to top":
        primary = -float(region.top)
    elif sort_order == "left to right":
        primary = float(region.left)
    elif sort_order == "right to left":
        primary = -float(region.left)
    else:
        raise ValueError(f"Unknown SEGS sort order: '{sort_order}'.")
    return primary, confidence_desc, region.top, region.left, index


def _coerce_header(value: object) -> SegsHeader:
    """Convert a SEGS header to an image shape tuple."""

    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("SEGS header must contain exactly height and width.")
    height = int(value[0])
    width = int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError("SEGS header height and width must be positive.")
    return height, width


def _looks_like_segs(value: object) -> bool:
    """Return whether a value has the outer shape of one SEGS payload."""

    if not isinstance(value, tuple) or len(value) != 2:
        return False
    return _looks_like_header(value[0]) and isinstance(value[1], Iterable)


def _looks_like_header(value: object) -> bool:
    """Return whether a value can represent a SEGS header."""

    if not isinstance(value, Sequence) or len(value) != 2:
        return False
    try:
        int(value[0])
        int(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _coerce_region(value: object, name: str) -> CropRegion:
    """Convert a four-value coordinate sequence into a crop region."""

    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"{name} must contain left, top, right, bottom.")
    region = CropRegion(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    _validate_region(region, name)
    return region


def _coerce_box(value: object, name: str) -> BoundingBox:
    """Convert a four-value coordinate sequence into a bounding box."""

    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"{name} must contain left, top, right, bottom.")
    box = BoundingBox(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    _validate_box(box, name)
    return box


def _validate_region(region: CropRegion, name: str) -> None:
    """Validate crop-region coordinate ordering."""

    if region.left < 0 or region.top < 0:
        raise ValueError(f"{name} left and top must be non-negative.")
    if region.right <= region.left or region.bottom <= region.top:
        raise ValueError(f"{name} right/bottom must be greater than left/top.")


def _validate_box(box: BoundingBox, name: str) -> None:
    """Validate bounding-box coordinate ordering."""

    if box.left < 0 or box.top < 0:
        raise ValueError(f"{name} left and top must be non-negative.")
    if box.right <= box.left or box.bottom <= box.top:
        raise ValueError(f"{name} right/bottom must be greater than left/top.")
