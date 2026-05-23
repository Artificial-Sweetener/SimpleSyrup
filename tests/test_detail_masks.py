# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for detailer-specific mask feathering."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.masking.detailer_masks import gaussian_feather_mask


def test_gaussian_feather_preserves_empty_and_full_masks() -> None:
    """Masks without an internal boundary remain stable after feathering."""

    empty = torch.zeros((6, 6), dtype=torch.float32)
    full = torch.ones((6, 6), dtype=torch.float32)

    assert torch.equal(gaussian_feather_mask(empty, 3), empty)
    assert torch.equal(gaussian_feather_mask(full, 3), full)


def test_gaussian_feather_softens_rectangular_mask_with_padding() -> None:
    """A padded rectangle gains a smooth alpha ramp around its boundary."""

    mask = torch.zeros((9, 9), dtype=torch.float32)
    mask[3:6, 3:6] = 1.0

    feathered = gaussian_feather_mask(mask, 2)

    assert feathered.shape == mask.shape
    assert 0.0 < float(feathered[2, 4]) < 1.0
    assert 0.0 < float(feathered[4, 2]) < 1.0
    assert float(feathered[4, 4]) > float(feathered[2, 4])
    assert float(feathered[0, 0]) < float(feathered[2, 4])


def test_gaussian_feather_preserves_bhw_shape_and_range() -> None:
    """Batched masks keep their shape and stay in normalized alpha range."""

    mask = torch.zeros((2, 7, 7), dtype=torch.float32)
    mask[:, 2:5, 2:5] = 1.0

    feathered = gaussian_feather_mask(mask, 2)

    assert feathered.shape == mask.shape
    assert float(feathered.min()) >= 0.0
    assert float(feathered.max()) <= 1.0
    assert 0.0 < float(feathered[0, 1, 3]) < 1.0


def test_gaussian_feather_skips_too_small_masks() -> None:
    """Masks too small for a useful blur are returned unchanged."""

    mask = torch.zeros((2, 2), dtype=torch.float32)
    mask[0, 0] = 1.0

    assert torch.equal(gaussian_feather_mask(mask, 3), mask)


def test_gaussian_feather_rejects_invalid_inputs() -> None:
    """Invalid mask inputs fail before producing misleading alpha masks."""

    with pytest.raises(TypeError, match="torch.Tensor"):
        gaussian_feather_mask([[1.0]], 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="HW or BHW"):
        gaussian_feather_mask(torch.zeros((1, 1, 2, 2)), 1)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        gaussian_feather_mask(torch.zeros((2, 2)), -1)
