# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Describe graph-visible full-canvas transformations for attention masks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

_ANCHORS = frozenset(
    {
        "center",
        "top-left",
        "top",
        "top-right",
        "left",
        "right",
        "bottom-left",
        "bottom",
        "bottom-right",
    }
)


class AttentionSpatialTransformKind(StrEnum):
    """Identify a supported mask-coordinate transformation."""

    RESIZE = "resize"
    FIT_RESIZE = "fit_resize"
    SCALE = "scale"
    COVER_CROP = "cover_crop"
    FIT_PAD = "fit_pad"


@dataclass(frozen=True, slots=True)
class AttentionSpatialTransform:
    """Hold validated resize, crop, or pad parameters from one graph node."""

    kind: AttentionSpatialTransformKind
    width: int | None = None
    height: int | None = None
    scale: float | None = None
    anchor: str = "center"
    divisible_by: int = 1

    def __post_init__(self) -> None:
        """Require complete parameters for the selected transformation kind."""

        if self.anchor not in _ANCHORS:
            raise ValueError("Attention spatial transform anchor is invalid.")
        if self.kind is AttentionSpatialTransformKind.SCALE:
            if self.scale is None or self.scale <= 0.0:
                raise ValueError("Attention scale transform requires a positive scale.")
            if self.width is not None or self.height is not None:
                raise ValueError("Attention scale transform cannot contain a size.")
            if self.divisible_by != 1:
                raise ValueError("Attention scale transform cannot set divisibility.")
            return
        if (
            type(self.width) is not int
            or self.width < 1
            or type(self.height) is not int
            or self.height < 1
            or self.scale is not None
        ):
            raise ValueError("Attention spatial transform requires a positive size.")
        if type(self.divisible_by) is not int or self.divisible_by < 1:
            raise ValueError("Attention spatial divisibility must be positive.")
