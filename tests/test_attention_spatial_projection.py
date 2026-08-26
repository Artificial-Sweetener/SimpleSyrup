# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test graph-visible attention mask coordinate projection."""

from __future__ import annotations

import torch

from simple_syrup.domain.attention_spatial_transform import (
    AttentionSpatialTransform,
    AttentionSpatialTransformKind,
)
from simple_syrup.services.attention_spatial_projection import (
    ATTENTION_SPATIAL_PROJECTION_SERVICE,
)


def test_cover_crop_projects_attention_with_requested_anchor() -> None:
    """Crop a covered map using the same anchor geometry as image resizing."""

    source = torch.tensor([[1.0, 0.75, 0.25, 0.0], [1.0, 0.75, 0.25, 0.0]])
    transform = AttentionSpatialTransform(
        AttentionSpatialTransformKind.COVER_CROP,
        width=2,
        height=2,
        anchor="center",
    )

    result = ATTENTION_SPATIAL_PROJECTION_SERVICE.project(source, (transform,))

    assert result.shape == (2, 2)
    assert torch.allclose(result, source[:, 1:3])


def test_fit_pad_projects_attention_without_filling_the_padded_canvas() -> None:
    """Keep padding outside the transformed source mask at zero alpha."""

    source = torch.ones(2, 4)
    transform = AttentionSpatialTransform(
        AttentionSpatialTransformKind.FIT_PAD,
        width=4,
        height=4,
        anchor="center",
    )

    result = ATTENTION_SPATIAL_PROJECTION_SERVICE.project(source, (transform,))

    assert result.shape == (4, 4)
    assert result[0].count_nonzero().item() == 0
    assert result[-1].count_nonzero().item() == 0
    assert result[1:3].eq(1.0).all().item()
