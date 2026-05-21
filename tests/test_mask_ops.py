# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for prompt mask composition helpers."""

from __future__ import annotations

import torch

from simple_syrup.masking.mask_ops import (
    MaskRefinementSettings,
    compose_prompt_masks,
    histogram_remap,
    refine_prompt_mask,
    zero_mask_like_image,
)


def test_zero_mask_like_image_matches_bhw_shape() -> None:
    """Zero mask helper uses image batch and spatial dimensions."""

    image = torch.ones((2, 4, 6, 3), dtype=torch.float32)

    mask = zero_mask_like_image(image)

    assert mask.shape == (2, 4, 6)
    assert torch.count_nonzero(mask) == 0


def test_compose_prompt_masks_returns_positive_when_negative_missing() -> None:
    """Positive masks are clamped when no negative mask is supplied."""

    positive = torch.tensor([[[1.2, 0.4]]], dtype=torch.float32)

    result = compose_prompt_masks(positive)

    assert torch.equal(result, torch.tensor([[[1.0, 0.4]]], dtype=torch.float32))


def test_compose_prompt_masks_subtracts_negative_and_clamps() -> None:
    """Negative masks remove matching positive mask regions."""

    positive = torch.tensor([[[1.0, 0.4, 0.2]]], dtype=torch.float32)
    negative = torch.tensor([[[0.25, 0.5, 0.0]]], dtype=torch.float32)

    result = compose_prompt_masks(positive, negative)

    assert torch.equal(result, torch.tensor([[[0.75, 0.0, 0.2]]], dtype=torch.float32))


def test_histogram_remap_uses_black_and_white_points() -> None:
    """Black and white points stretch mask values into the ComfyUI mask range."""

    mask = torch.tensor([[[0.15, 0.57, 0.99]]], dtype=torch.float32)

    result = histogram_remap(mask, black_point=0.15, white_point=0.99)

    assert torch.allclose(result, torch.tensor([[[0.0, 0.5, 1.0]]]))


def test_refine_prompt_mask_preserves_shape_with_size_limit() -> None:
    """Detail refinement keeps the original BHW mask shape after bounded work."""

    image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
    mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mask[:, 2:6, 2:6] = 1.0

    result = refine_prompt_mask(
        mask,
        image,
        MaskRefinementSettings(
            detail_method="PyMatting",
            detail_erode=2,
            detail_dilate=2,
            black_point=0.0,
            white_point=1.0,
            process_detail=True,
            execution_device="cpu",
            max_size_pixels=16,
        ),
    )

    assert result.shape == mask.shape
    assert result.dtype == torch.float32
    assert torch.all((0.0 <= result) & (result <= 1.0))


def test_refine_prompt_mask_rejects_invalid_points() -> None:
    """Invalid level remap settings fail before processing."""

    image = torch.rand((1, 2, 2, 3), dtype=torch.float32)
    mask = torch.ones((1, 2, 2), dtype=torch.float32)

    try:
        refine_prompt_mask(
            mask,
            image,
            MaskRefinementSettings(
                detail_method="GuidedFilter",
                detail_erode=1,
                detail_dilate=1,
                black_point=0.8,
                white_point=0.2,
                process_detail=False,
                execution_device="cpu",
                max_size_pixels=4,
            ),
        )
    except ValueError as error:
        assert "black_point and white_point" in str(error)
    else:
        raise AssertionError("Expected invalid black/white points to fail.")
