# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove the complete standard-UNet regional output geometry matrix."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.standard_unet_variant_masks import (
    StandardUnetVariantMaskProjector,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_output import (
    StandardUnetVariantOutputComposer,
)


def test_hard_feather_overlap_and_uncovered_pixels_preserve_authored_math() -> None:
    """Select, blend, normalize, and fill each canonical coverage class."""

    values = torch.tensor(
        [
            [[1.0, 0.25, 0.75, 0.0]],
            [[0.0, 0.0, 0.75, 0.0]],
        ]
    )
    bank = RegionalMaskBank(values.clone(), values.clone(), 4, 1)
    model_input = torch.zeros((1, 1, 1, 4))
    masks = StandardUnetVariantMaskProjector().project(
        bank=bank,
        region_indices=(0, 1),
        model_input=model_input,
        transformer_options={},
    )

    result = StandardUnetVariantOutputComposer.compose(
        (
            torch.full_like(model_input, 2.0),
            torch.full_like(model_input, 4.0),
        ),
        masks,
        base_output=torch.full_like(model_input, 10.0),
    )

    assert StandardUnetVariantMaskProjector.requires_base(bank, (0, 1))
    torch.testing.assert_close(
        result[0, 0, 0],
        torch.tensor([2.0, 8.0, 3.0, 10.0]),
        rtol=0,
        atol=0,
    )
