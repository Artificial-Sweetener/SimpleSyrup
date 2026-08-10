# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for optional regional sampler input preparation."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.services.regional_sampling_preparation_service import (
    RegionalSamplingPreparationService,
)


def test_preparation_is_inactive_without_both_masks_and_a_batch() -> None:
    """Partial regional inputs preserve ordinary sampler interpretation."""

    service = RegionalSamplingPreparationService()
    latent = {"samples": torch.zeros((1, 4, 2, 4))}

    without_masks = service.prepare(
        positive=ConditioningBatch((_conditioning("global"),)),
        negative=_conditioning("negative"),
        latent_image=latent,
        region_masks=None,
        regional_prompt_weight=0.5,
        region_mask_feather=0,
    )
    without_batch = service.prepare(
        positive=_conditioning("global"),
        negative=_conditioning("negative"),
        latent_image=latent,
        region_masks=object(),
        regional_prompt_weight=0.5,
        region_mask_feather=0,
    )

    assert not without_masks.active
    assert isinstance(without_masks.positive, ConditioningBatch)
    assert not without_batch.active
    assert without_batch.planning_masks is None


def test_preparation_assembles_global_first_batch_at_latent_size() -> None:
    """Active regional sampling shares one resized mask batch with its planner."""

    masks = torch.zeros((2, 4, 8))
    masks[0, :, :4] = 1.0
    masks[1, :, 4:] = 1.0

    prepared = RegionalSamplingPreparationService().prepare(
        positive=ConditioningBatch(
            (
                _conditioning("global"),
                _conditioning("left"),
                _conditioning("right"),
            )
        ),
        negative=_conditioning("negative"),
        latent_image={"samples": torch.zeros((1, 4, 2, 4))},
        region_masks=masks,
        regional_prompt_weight=0.75,
        region_mask_feather=0,
    )

    assert prepared.active
    assert prepared.planning_masks is not None
    assert prepared.planning_masks.shape == (2, 2, 4)
    assert isinstance(prepared.positive, list)
    assert [item[0] for item in prepared.positive] == ["global", "left", "right"]
    assert prepared.positive[1][1]["mask"].shape == (1, 2, 4)
    assert prepared.positive[1][1]["mask_strength"] == 0.75


def test_preparation_reuses_prompt_by_region_mismatch_policy() -> None:
    """Excess regional prompts fail through the authoritative pairing service."""

    with pytest.raises(ValueError, match="2 regional entries but only 1 authored"):
        RegionalSamplingPreparationService().prepare(
            positive=ConditioningBatch(
                (
                    _conditioning("global"),
                    _conditioning("first"),
                    _conditioning("excess"),
                )
            ),
            negative=_conditioning("negative"),
            latent_image={"samples": torch.zeros((1, 4, 2, 4))},
            region_masks=torch.ones((1, 4, 8)),
            regional_prompt_weight=0.5,
            region_mask_feather=0,
        )


def _conditioning(name: str) -> list[list[object]]:
    """Return one structurally valid standard conditioning value."""

    return [[name, {}]]
