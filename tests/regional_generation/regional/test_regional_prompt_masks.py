# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for full-context regional prompt mask preparation."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.masking.regional_prompt_masks import (
    prepare_regional_mask_batch,
    regional_mask,
    resize_regional_mask_batch,
)


def test_prepare_regional_mask_batch_clamps_without_mutating_input() -> None:
    """Mask preparation normalizes authored values on a separate tensor."""

    masks = torch.tensor([[[-1.0, 0.5, 2.0]]])
    original = masks.clone()

    prepared = prepare_regional_mask_batch(masks, feather=0)

    assert torch.equal(prepared, torch.tensor([[[0.0, 0.5, 1.0]]]))
    assert torch.equal(masks, original)


def test_regional_mask_returns_one_ordered_batch_entry() -> None:
    """Positional selection preserves authored mask order and BHW shape."""

    masks = torch.stack([torch.zeros((2, 2)), torch.ones((2, 2))])

    selected = regional_mask(masks, 1)

    assert selected.shape == (1, 2, 2)
    assert torch.equal(selected, masks[1:2])


def test_regional_mask_rejects_out_of_range_index() -> None:
    """Selection fails explicitly when prompt and mask planning diverge."""

    with pytest.raises(IndexError, match="out of range"):
        regional_mask(torch.ones((1, 2, 2)), 1)


def test_mask_downscale_preserves_one_pixel_as_fractional_latent_coverage() -> None:
    """Retain sub-token authored area instead of missing it between sample centers."""

    source = torch.zeros((1, 1536, 1536))
    source[:, :, 0] = 1.0

    resized = resize_regional_mask_batch(source, height=192, width=192)

    assert torch.allclose(resized[:, :, 0], torch.full((1, 192), 0.125))
    assert torch.count_nonzero(resized) == 192
    assert float(resized.mean()) == pytest.approx(float(source.mean()))
