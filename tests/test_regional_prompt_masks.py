# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for full-context regional prompt mask preparation."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.masking.regional_prompt_masks import (
    complementary_global_prompt_mask,
)


def test_complementary_global_mask_uses_clamped_accumulated_coverage() -> None:
    """Overlapping regions accumulate while global coverage remains normalized."""

    masks = torch.tensor(
        [
            [[1.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0]],
        ]
    )

    global_mask = complementary_global_prompt_mask(masks, (0, 1), 0.5)
    regional_total = masks.sum(dim=0, keepdim=True) * 0.5
    global_share = global_mask / (global_mask + regional_total)

    assert torch.equal(global_mask, torch.tensor([[[0.5, 0.5, 0.5]]]))
    assert torch.allclose(global_share, torch.tensor([[[0.5, 1.0 / 3.0, 0.5]]]))


def test_complementary_global_mask_uses_only_paired_indices() -> None:
    """Extra authored masks do not reduce global prompt influence."""

    masks = torch.stack([torch.zeros((2, 2)), torch.ones((2, 2))])

    global_mask = complementary_global_prompt_mask(masks, (0,), 1.0)

    assert torch.equal(global_mask, torch.ones((1, 2, 2)))


def test_complementary_global_mask_requires_a_regional_pair() -> None:
    """Coverage cannot be calculated without a paired regional prompt."""

    with pytest.raises(ValueError, match="at least one regional mask"):
        complementary_global_prompt_mask(torch.ones((1, 2, 2)), (), 0.5)
