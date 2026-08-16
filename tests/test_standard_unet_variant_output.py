# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify persistent standard-UNet output mask projection and composition."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.standard_unet_variant_masks import (
    StandardUnetVariantMaskProjector,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_output import (
    StandardUnetVariantOutputComposer,
)


def test_full_exhaustive_masks_compose_without_base_execution() -> None:
    """Select hard regional predictions without a hidden global branch."""

    bank = _bank(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
    model_input = torch.zeros((2, 4, 1, 2))
    masks = StandardUnetVariantMaskProjector().project(
        bank=bank,
        region_indices=(0, 1),
        model_input=model_input,
        transformer_options={},
    )

    result = StandardUnetVariantOutputComposer.compose(
        (torch.ones_like(model_input), torch.full_like(model_input, 3.0)),
        masks,
        base_output=None,
    )

    assert not StandardUnetVariantMaskProjector.requires_base(bank, (0, 1))
    assert torch.equal(result[..., 0], torch.ones_like(result[..., 0]))
    assert torch.equal(result[..., 1], torch.full_like(result[..., 1], 3.0))


def test_feather_overlap_normalizes_and_uncovered_pixels_use_base() -> None:
    """Interpolate gaps with base and average overlap without amplification."""

    bank = _bank(torch.tensor([[[0.5, 1.0, 0.0]], [[0.0, 1.0, 0.0]]]))
    model_input = torch.zeros((1, 4, 1, 3))
    masks = StandardUnetVariantMaskProjector().project(
        bank=bank,
        region_indices=(0, 1),
        model_input=model_input,
        transformer_options={},
    )

    result = StandardUnetVariantOutputComposer.compose(
        (torch.full_like(model_input, 2.0), torch.full_like(model_input, 4.0)),
        masks,
        base_output=torch.zeros_like(model_input),
    )

    assert StandardUnetVariantMaskProjector.requires_base(bank, (0, 1))
    assert torch.equal(result[0, 0, 0], torch.tensor([1.0, 3.0, 0.0]))


def _bank(masks: torch.Tensor) -> RegionalMaskBank:
    """Return one canonical bank with independent mask storage."""

    return RegionalMaskBank(
        masks.clone(),
        masks.clone(),
        int(masks.shape[-1]),
        int(masks.shape[-2]),
    )
