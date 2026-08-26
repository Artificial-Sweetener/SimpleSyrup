# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project captured attention through graph-visible full-canvas transforms."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ..domain.attention_spatial_transform import (
    AttentionSpatialTransform,
    AttentionSpatialTransformKind,
)
from ..image.resize_geometry import (
    CropPosition,
    ResizeMode,
    ResizeTarget,
    build_resize_plan,
)


class AttentionSpatialProjectionService:
    """Apply ordered resize, crop, and pad geometry to one attention map."""

    def project(
        self,
        values: torch.Tensor,
        transforms: tuple[AttentionSpatialTransform, ...],
    ) -> torch.Tensor:
        """Return one HW tensor transformed in graph-forward order."""

        result = values.float()
        for transform in transforms:
            result = _apply_transform(result, transform)
        return result


def _apply_transform(
    values: torch.Tensor, transform: AttentionSpatialTransform
) -> torch.Tensor:
    """Apply one validated transform using the product resize geometry policy."""

    if transform.kind is AttentionSpatialTransformKind.SCALE:
        if transform.scale is None:
            raise ValueError("Attention scale transform is incomplete.")
        height = max(1, round(int(values.shape[0]) * transform.scale))
        width = max(1, round(int(values.shape[1]) * transform.scale))
        return _resize(values, height, width)
    if transform.width is None or transform.height is None:
        raise ValueError("Attention sized transform is incomplete.")
    mode = {
        AttentionSpatialTransformKind.RESIZE: ResizeMode.STRETCH,
        AttentionSpatialTransformKind.FIT_RESIZE: ResizeMode.KEEP_AR,
        AttentionSpatialTransformKind.COVER_CROP: ResizeMode.CROP,
        AttentionSpatialTransformKind.FIT_PAD: ResizeMode.PAD,
    }[transform.kind]
    plan = build_resize_plan(
        int(values.shape[1]),
        int(values.shape[0]),
        ResizeTarget(transform.width, transform.height, transform.divisible_by),
        mode,
        CropPosition(transform.anchor),
    )
    resized = _resize(values, plan.resize_height, plan.resize_width)
    if plan.has_crop:
        resized = resized[
            plan.crop_y : plan.crop_y + plan.output_height,
            plan.crop_x : plan.crop_x + plan.output_width,
        ]
    if plan.has_pad:
        resized = functional.pad(
            resized,
            (plan.pad_left, plan.pad_right, plan.pad_top, plan.pad_bottom),
            value=0.0,
        )
    return resized


def _resize(values: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize one HW map with stable bilinear interpolation."""

    return functional.interpolate(
        values.reshape(1, 1, int(values.shape[0]), int(values.shape[1])),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


ATTENTION_SPATIAL_PROJECTION_SERVICE = AttentionSpatialProjectionService()
