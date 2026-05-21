# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure geometry planning for target image resizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class ResizeMode(StrEnum):
    """Supported target resize modes."""

    STRETCH = "Stretch"
    KEEP_AR = "Keep AR"
    CROP = "Crop (Cover + Crop)"
    PAD = "Pad (Fit + Pad)"


class CropPosition(StrEnum):
    """Supported crop and pad anchor positions."""

    CENTER = "center"
    TOP_LEFT = "top-left"
    TOP = "top"
    TOP_RIGHT = "top-right"
    LEFT = "left"
    RIGHT = "right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM = "bottom"
    BOTTOM_RIGHT = "bottom-right"


@dataclass(frozen=True)
class ResizeTarget:
    """Requested output bounds and divisibility constraint."""

    width: int
    height: int
    divisible_by: int = 1


@dataclass(frozen=True)
class ResizePlan:
    """Concrete resize, crop, and pad geometry for an image batch."""

    resize_width: int
    resize_height: int
    output_width: int
    output_height: int
    crop_x: int = 0
    crop_y: int = 0
    pad_left: int = 0
    pad_right: int = 0
    pad_top: int = 0
    pad_bottom: int = 0

    @property
    def has_crop(self) -> bool:
        """Return whether the plan crops after resizing."""

        return self.crop_x > 0 or self.crop_y > 0

    @property
    def has_pad(self) -> bool:
        """Return whether the plan pads after resizing."""

        return any(
            side > 0
            for side in (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
        )


def build_resize_plan(
    source_width: int,
    source_height: int,
    target: ResizeTarget,
    mode: ResizeMode | str,
    position: CropPosition | str,
) -> ResizePlan:
    """Build resize geometry for the requested mode and source dimensions."""

    _validate_positive_dimension(source_width, "source_width")
    _validate_positive_dimension(source_height, "source_height")
    _validate_positive_dimension(target.width, "target.width")
    _validate_positive_dimension(target.height, "target.height")

    normalized_mode = _coerce_resize_mode(mode)
    normalized_position = _coerce_crop_position(position)
    divisible_by = _normalize_divisible_by(target.divisible_by)

    if normalized_mode is ResizeMode.STRETCH:
        output_width, output_height = apply_divisibility(
            target.width,
            target.height,
            divisible_by,
        )
        return ResizePlan(
            resize_width=output_width,
            resize_height=output_height,
            output_width=output_width,
            output_height=output_height,
        )

    if normalized_mode is ResizeMode.KEEP_AR:
        resize_width, resize_height = fit_inside(
            source_width,
            source_height,
            target.width,
            target.height,
        )
        output_width, output_height = apply_divisibility(
            resize_width,
            resize_height,
            divisible_by,
        )
        return ResizePlan(
            resize_width=output_width,
            resize_height=output_height,
            output_width=output_width,
            output_height=output_height,
        )

    output_width, output_height = apply_divisibility(
        target.width,
        target.height,
        divisible_by,
    )

    if normalized_mode is ResizeMode.CROP:
        resize_width, resize_height = cover_bounds(
            source_width,
            source_height,
            output_width,
            output_height,
        )
        crop_x, crop_y = calculate_crop_offsets(
            normalized_position,
            resize_width,
            resize_height,
            output_width,
            output_height,
        )
        return ResizePlan(
            resize_width=resize_width,
            resize_height=resize_height,
            output_width=output_width,
            output_height=output_height,
            crop_x=crop_x,
            crop_y=crop_y,
        )

    if normalized_mode is ResizeMode.PAD:
        resize_width, resize_height = fit_inside(
            source_width,
            source_height,
            output_width,
            output_height,
        )
        pad_width = output_width - resize_width
        pad_height = output_height - resize_height
        pad_left, pad_right, pad_top, pad_bottom = calculate_pad_sides(
            normalized_position,
            pad_width,
            pad_height,
        )
        return ResizePlan(
            resize_width=resize_width,
            resize_height=resize_height,
            output_width=output_width,
            output_height=output_height,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
        )

    raise ValueError(f"Unsupported resize mode: {mode!r}")


def fit_inside(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """Return dimensions that fit inside target bounds while preserving aspect."""

    _validate_positive_dimension(source_width, "source_width")
    _validate_positive_dimension(source_height, "source_height")
    _validate_positive_dimension(target_width, "target_width")
    _validate_positive_dimension(target_height, "target_height")

    scale = min(target_width / source_width, target_height / source_height)
    return (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )


def cover_bounds(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """Return dimensions that cover target bounds while preserving aspect."""

    _validate_positive_dimension(source_width, "source_width")
    _validate_positive_dimension(source_height, "source_height")
    _validate_positive_dimension(target_width, "target_width")
    _validate_positive_dimension(target_height, "target_height")

    scale = max(target_width / source_width, target_height / source_height)
    return (
        max(1, int(math.ceil(source_width * scale))),
        max(1, int(math.ceil(source_height * scale))),
    )


def apply_divisibility(
    width: int,
    height: int,
    divisible_by: int,
) -> tuple[int, int]:
    """Step dimensions down to positive multiples of the divisibility value."""

    _validate_positive_dimension(width, "width")
    _validate_positive_dimension(height, "height")
    normalized = _normalize_divisible_by(divisible_by)

    if normalized <= 1:
        return width, height

    output_width = width - (width % normalized)
    output_height = height - (height % normalized)
    if output_width <= 0 or output_height <= 0:
        raise ValueError(
            "Requested dimensions cannot satisfy divisible_by="
            f"{normalized}: width={width}, height={height}."
        )
    return output_width, output_height


def calculate_crop_offsets(
    position: CropPosition | str,
    resized_width: int,
    resized_height: int,
    output_width: int,
    output_height: int,
) -> tuple[int, int]:
    """Return the x and y offsets for cropping resized content."""

    normalized_position = _coerce_crop_position(position)
    _validate_positive_dimension(resized_width, "resized_width")
    _validate_positive_dimension(resized_height, "resized_height")
    _validate_positive_dimension(output_width, "output_width")
    _validate_positive_dimension(output_height, "output_height")

    if resized_width < output_width or resized_height < output_height:
        raise ValueError(
            "Crop dimensions must be at least as large as output dimensions: "
            f"resized={resized_width}x{resized_height}, "
            f"output={output_width}x{output_height}."
        )

    extra_width = resized_width - output_width
    extra_height = resized_height - output_height
    return (
        _offset_for_axis(normalized_position, extra_width, horizontal=True),
        _offset_for_axis(normalized_position, extra_height, horizontal=False),
    )


def calculate_pad_sides(
    position: CropPosition | str,
    pad_width: int,
    pad_height: int,
) -> tuple[int, int, int, int]:
    """Return left, right, top, and bottom padding for an anchor position."""

    normalized_position = _coerce_crop_position(position)
    if pad_width < 0 or pad_height < 0:
        raise ValueError(
            f"Padding cannot be negative: pad_width={pad_width}, "
            f"pad_height={pad_height}."
        )

    left = _offset_for_axis(normalized_position, pad_width, horizontal=True)
    top = _offset_for_axis(normalized_position, pad_height, horizontal=False)
    right = pad_width - left
    bottom = pad_height - top
    return left, right, top, bottom


def _offset_for_axis(
    position: CropPosition,
    extra: int,
    *,
    horizontal: bool,
) -> int:
    """Resolve a crop or pad offset along one axis."""

    if extra <= 0:
        return 0

    if horizontal:
        if position in {
            CropPosition.TOP_LEFT,
            CropPosition.LEFT,
            CropPosition.BOTTOM_LEFT,
        }:
            return 0
        if position in {
            CropPosition.TOP_RIGHT,
            CropPosition.RIGHT,
            CropPosition.BOTTOM_RIGHT,
        }:
            return extra
        return extra // 2

    if position in {
        CropPosition.TOP_LEFT,
        CropPosition.TOP,
        CropPosition.TOP_RIGHT,
    }:
        return 0
    if position in {
        CropPosition.BOTTOM_LEFT,
        CropPosition.BOTTOM,
        CropPosition.BOTTOM_RIGHT,
    }:
        return extra
    return extra // 2


def _coerce_resize_mode(mode: ResizeMode | str) -> ResizeMode:
    """Convert a raw resize mode value into a supported enum."""

    try:
        return mode if isinstance(mode, ResizeMode) else ResizeMode(str(mode))
    except ValueError as exc:
        raise ValueError(f"Unsupported resize mode: {mode!r}") from exc


def _coerce_crop_position(position: CropPosition | str) -> CropPosition:
    """Convert a raw crop position value into a supported enum."""

    try:
        return (
            position
            if isinstance(position, CropPosition)
            else CropPosition(str(position))
        )
    except ValueError as exc:
        raise ValueError(f"Unsupported crop_position: {position!r}") from exc


def _normalize_divisible_by(divisible_by: int) -> int:
    """Validate and normalize the divisibility constraint."""

    normalized = int(divisible_by)
    if normalized < 1:
        raise ValueError(f"divisible_by must be at least 1, got {divisible_by}.")
    return normalized


def _validate_positive_dimension(value: int, name: str) -> None:
    """Validate a positive integer dimension."""

    if int(value) <= 0:
        raise ValueError(f"{name} must be greater than 0, got {value}.")
