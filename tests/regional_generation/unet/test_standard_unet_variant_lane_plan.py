# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exhaustive and genuine-gap standard-UNet lane policy."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.standard_unet_variant_lane_plan import (
    StandardUnetVariantLanePlanner,
)


def test_exhaustive_inactive_region_becomes_regional_base_lane() -> None:
    """Use native weights under phased region-owned graph execution."""

    plan = StandardUnetVariantLanePlanner().plan(_exhaustive_bank(), (0,))

    assert plan.regional_base_indices == (1,)
    assert plan.requires_global_base is False
    assert plan.output_region_indices == (0, 1)


def test_genuine_gap_retains_global_base_without_regional_base_lanes() -> None:
    """Keep one globally composed base for any unauthored canvas pixel."""

    masks = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
    bank = RegionalMaskBank(masks.clone(), masks.clone(), 3, 1)

    plan = StandardUnetVariantLanePlanner().plan(bank, (0,))

    assert plan.regional_base_indices == ()
    assert plan.requires_global_base is True
    assert plan.output_region_indices == (0,)


def test_fully_active_exhaustive_regions_need_no_base_lane() -> None:
    """Preserve the accepted variant-only topology when every region is active."""

    plan = StandardUnetVariantLanePlanner().plan(_exhaustive_bank(), (0, 1))

    assert plan.regional_base_indices == ()
    assert plan.requires_global_base is False
    assert plan.output_region_indices == (0, 1)


def _exhaustive_bank() -> RegionalMaskBank:
    """Return two hard masks covering every canonical pixel exactly once."""

    masks = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    return RegionalMaskBank(masks.clone(), masks.clone(), 2, 1)
