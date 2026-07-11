# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for converting existing masks into image-associated SEGS."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from simple_syrup.domain.segs import CropRegion
from simple_syrup.services.mask_to_segs_service import MaskToSEGSService


def test_rectangular_mask_creates_image_associated_seg() -> None:
    """A single active mask region becomes one SEG with matching image crop."""

    image = _image()
    mask = torch.zeros((6, 6), dtype=torch.float32)
    mask[1:4, 2:5] = 1.0

    header, segments = MaskToSEGSService().build(
        image=image,
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        label="paint",
    )

    assert header == (6, 6)
    assert len(segments) == 1
    segment = segments[0]
    assert segment.crop_region == CropRegion(2, 1, 5, 4)
    assert segment.bbox == (2, 1, 5, 4)
    assert segment.label == "paint"
    assert segment.confidence == 1.0
    assert torch.equal(cast(torch.Tensor, segment.cropped_image), image[:, 1:4, 2:5])
    assert torch.equal(cast(torch.Tensor, segment.cropped_mask), torch.ones((3, 3)))


def test_disconnected_regions_create_separate_segs() -> None:
    """Separated mask islands become separate SEGS in scan order."""

    mask = torch.zeros((6, 6), dtype=torch.float32)
    mask[0:2, 0:2] = 1.0
    mask[4:6, 4:6] = 1.0

    _header, segments = MaskToSEGSService().build(
        image=_image(),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        label="mask",
    )

    assert [segment.crop_region for segment in segments] == [
        CropRegion(0, 0, 2, 2),
        CropRegion(4, 4, 6, 6),
    ]


def test_diagonal_contact_uses_eight_connected_components() -> None:
    """Diagonally touching pixels are treated as one painted region."""

    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1, 1] = 1.0
    mask[2, 2] = 1.0

    _header, segments = MaskToSEGSService().build(
        image=torch.zeros((1, 4, 4, 3), dtype=torch.float32),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        label="mask",
    )

    assert len(segments) == 1
    assert segments[0].bbox == (1, 1, 3, 3)


def test_empty_mask_returns_empty_native_segs() -> None:
    """Empty masks produce an empty native SEGS payload."""

    header, segments = MaskToSEGSService().build(
        image=_image(),
        mask=torch.zeros((6, 6), dtype=torch.float32),
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        label="mask",
    )

    assert header == (6, 6)
    assert segments == ()


def test_mask_threshold_controls_active_pixels() -> None:
    """Soft pixels below threshold do not become SEG regions."""

    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1, 1] = 0.4
    mask[2, 2] = 0.6

    _header, segments = MaskToSEGSService().build(
        image=torch.zeros((1, 4, 4, 3), dtype=torch.float32),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        label="mask",
    )

    assert len(segments) == 1
    assert segments[0].bbox == (2, 2, 3, 3)
    assert cast(torch.Tensor, segments[0].cropped_mask).item() == pytest.approx(0.6)


def test_mask_dilation_grows_source_mask_before_extraction() -> None:
    """Positive mask dilation expands topology before bounding boxes are built."""

    mask = torch.zeros((7, 7), dtype=torch.float32)
    mask[3, 3] = 1.0

    _header, segments = MaskToSEGSService().build(
        image=torch.zeros((1, 7, 7, 3), dtype=torch.float32),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=3,
        post_dilation=0,
        crop_factor=1.0,
        label="mask",
    )

    assert segments[0].bbox == (2, 2, 5, 5)
    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 9.0


def test_negative_mask_dilation_erodes_source_mask() -> None:
    """Negative mask dilation shrinks active regions before extraction."""

    mask = torch.zeros((7, 7), dtype=torch.float32)
    mask[1:6, 1:6] = 1.0

    _header, segments = MaskToSEGSService().build(
        image=torch.zeros((1, 7, 7, 3), dtype=torch.float32),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=-3,
        post_dilation=0,
        crop_factor=1.0,
        label="mask",
    )

    assert segments[0].bbox == (2, 2, 5, 5)
    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 9.0


def test_post_dilation_changes_crop_local_mask_after_crop() -> None:
    """Post dilation applies to each final cropped SEG mask."""

    mask = torch.zeros((7, 7), dtype=torch.float32)
    mask[2:5, 2:5] = 1.0

    _header, segments = MaskToSEGSService().build(
        image=torch.zeros((1, 7, 7, 3), dtype=torch.float32),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=-3,
        crop_factor=3.0,
        label="mask",
    )

    assert cast(torch.Tensor, segments[0].cropped_mask).sum().item() == 1.0


def test_crop_factor_expands_crop_region() -> None:
    """Crop factor adds image context around the extracted mask bbox."""

    mask = torch.zeros((8, 8), dtype=torch.float32)
    mask[3:5, 3:5] = 1.0

    _header, segments = MaskToSEGSService().build(
        image=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=3.0,
        label="mask",
    )

    assert segments[0].crop_region == CropRegion(1, 1, 7, 7)
    assert cast(torch.Tensor, segments[0].cropped_mask).shape == (6, 6)


def test_crop_factor_zero_uses_full_image() -> None:
    """A zero crop factor selects the full image as the SEG crop."""

    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1:3, 1:3] = 1.0

    _header, segments = MaskToSEGSService().build(
        image=torch.zeros((1, 4, 4, 3), dtype=torch.float32),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=0.0,
        label="mask",
    )

    assert segments[0].crop_region == CropRegion(0, 0, 4, 4)
    assert cast(torch.Tensor, segments[0].cropped_mask).shape == (4, 4)


def test_fractional_crop_factor_below_one_fails() -> None:
    """Crop factor accepts zero or values at least one, but not in between."""

    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1:3, 1:3] = 1.0

    with pytest.raises(ValueError, match="crop_factor"):
        MaskToSEGSService().build(
            image=torch.zeros((1, 4, 4, 3), dtype=torch.float32),
            mask=mask,
            mask_threshold=0.5,
            size_threshold=1,
            mask_dilation=0,
            post_dilation=0,
            crop_factor=0.5,
            label="mask",
        )


def test_size_threshold_drops_small_components() -> None:
    """Regions below the minimum bbox size are discarded."""

    mask = torch.zeros((6, 6), dtype=torch.float32)
    mask[1:3, 1:3] = 1.0

    _header, segments = MaskToSEGSService().build(
        image=_image(),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=3,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        label="mask",
    )

    assert segments == ()


def test_soft_mask_values_are_retained_inside_component() -> None:
    """SEG masks preserve source mask strength inside extracted components."""

    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1, 1] = 0.75
    mask[1, 2] = 0.5

    _header, segments = MaskToSEGSService().build(
        image=torch.zeros((1, 4, 4, 3), dtype=torch.float32),
        mask=mask,
        mask_threshold=0.5,
        size_threshold=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        label="mask",
    )

    assert torch.equal(
        cast(torch.Tensor, segments[0].cropped_mask),
        torch.tensor([[0.75, 0.5]], dtype=torch.float32),
    )


def _image() -> torch.Tensor:
    """Return a deterministic 6x6 test image."""

    return torch.arange(1 * 6 * 6 * 3, dtype=torch.float32).reshape(1, 6, 6, 3) / 255
