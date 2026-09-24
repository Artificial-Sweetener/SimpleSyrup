# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the immutable full-canvas regional mask authority."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.masking.regional_prompt_masks import build_regional_mask_bank


def test_factory_builds_separate_full_canvas_mask_forms() -> None:
    """Preserve authored geometry while projecting both forms exactly once."""

    source = torch.zeros((2, 4, 8))
    source[0, :, :4] = 1.0
    source[1, :, 4:] = 1.0
    original = source.clone()

    bank = build_regional_mask_bank(
        source,
        feather=0,
        canvas_height=2,
        canvas_width=4,
    )

    assert bank.region_count == 2
    assert bank.planning_masks.shape == (2, 2, 4)
    assert bank.conditioning_masks.shape == (2, 2, 4)
    assert torch.equal(bank.planning_masks, bank.conditioning_masks)
    assert (
        bank.planning_masks.untyped_storage().data_ptr()
        != bank.conditioning_masks.untyped_storage().data_ptr()
    )
    assert torch.equal(source, original)


def test_factory_keeps_feathering_out_of_planning_geometry() -> None:
    """Keep hard authored coverage independent from soft conditioning influence."""

    source = torch.zeros((1, 5, 5))
    source[:, 2, 2] = 1.0

    bank = build_regional_mask_bank(
        source,
        feather=1,
        canvas_height=5,
        canvas_width=5,
    )

    assert torch.equal(bank.planning_masks, source)
    assert not torch.equal(bank.conditioning_masks, source)
    assert bool((bank.conditioning_masks > 0.0).sum() > 1)


def test_regional_mask_bank_fields_are_frozen() -> None:
    """Prevent replacement of canonical mask tensors or canvas geometry."""

    bank = _bank()
    canvas_attribute = "canvas_width"

    with pytest.raises(FrozenInstanceError):
        setattr(bank, canvas_attribute, 8)


@pytest.mark.parametrize(
    ("planning", "conditioning", "width", "height", "message"),
    [
        (
            cast(Any, "invalid"),
            torch.ones((1, 2, 4)),
            4,
            2,
            "Planning regional masks must be a torch.Tensor",
        ),
        (
            torch.ones((2, 4)),
            torch.ones((2, 4)).clone(),
            4,
            2,
            "Planning regional masks must use BHW layout",
        ),
        (
            torch.ones((0, 2, 4)),
            torch.ones((0, 2, 4)).clone(),
            4,
            2,
            "at least one region",
        ),
        (
            torch.ones((1, 2, 3)),
            torch.ones((1, 2, 3)).clone(),
            4,
            2,
            "match the full latent canvas 4x2",
        ),
        (
            torch.ones((1, 2, 4), dtype=torch.int64),
            torch.ones((1, 2, 4), dtype=torch.int64).clone(),
            4,
            2,
            "floating-point dtype",
        ),
        (
            torch.tensor([[[float("nan"), 0.0, 0.0, 0.0], [0.0] * 4]]),
            torch.zeros((1, 2, 4)),
            4,
            2,
            "only finite values",
        ),
        (
            torch.tensor([[[1.1, 0.0, 0.0, 0.0], [0.0] * 4]]),
            torch.zeros((1, 2, 4)),
            4,
            2,
            r"stay within \[0, 1\]",
        ),
    ],
)
def test_regional_mask_bank_rejects_malformed_authority(
    planning: torch.Tensor,
    conditioning: torch.Tensor,
    width: int,
    height: int,
    message: str,
) -> None:
    """Fail closed for malformed canonical tensor and canvas state."""

    with pytest.raises((TypeError, ValueError), match=message):
        RegionalMaskBank(
            planning_masks=planning,
            conditioning_masks=conditioning,
            canvas_width=width,
            canvas_height=height,
        )


def test_regional_mask_bank_rejects_divergent_dtype() -> None:
    """Require both mask forms to share one execution dtype."""

    with pytest.raises(ValueError, match="mask dtypes must match"):
        RegionalMaskBank(
            planning_masks=torch.ones((1, 2, 4), dtype=torch.float32),
            conditioning_masks=torch.ones((1, 2, 4), dtype=torch.float64),
            canvas_width=4,
            canvas_height=2,
        )


def test_regional_mask_bank_rejects_shared_tensor_storage() -> None:
    """Prevent conditioning edits from mutating authored planning geometry."""

    masks = torch.ones((1, 2, 4))

    with pytest.raises(ValueError, match="must not share storage"):
        RegionalMaskBank(
            planning_masks=masks,
            conditioning_masks=masks.view_as(masks),
            canvas_width=4,
            canvas_height=2,
        )


def test_regional_mask_bank_rejects_invalid_canvas() -> None:
    """Require a positive full-canvas authority before tensor validation."""

    with pytest.raises(ValueError, match="canvas dimensions must be positive"):
        RegionalMaskBank(
            planning_masks=torch.ones((1, 2, 4)),
            conditioning_masks=torch.ones((1, 2, 4)).clone(),
            canvas_width=0,
            canvas_height=2,
        )


def _bank() -> RegionalMaskBank:
    """Return one valid canonical regional mask bank."""

    return RegionalMaskBank(
        planning_masks=torch.ones((1, 2, 4)),
        conditioning_masks=torch.ones((1, 2, 4)).clone(),
        canvas_width=4,
        canvas_height=2,
    )
