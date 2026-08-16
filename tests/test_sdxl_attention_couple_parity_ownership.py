# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize exact reference and candidate parity mask ownership."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.attention_coupling.standard_unet_attention_weighting import (
    StandardUnetAttentionWeightingPolicy,
)
from tools.sdxl_attention_couple_parity.ownership import (
    REGIONAL_OWNERSHIP_STRENGTH,
    reference_base_mask,
)


@pytest.mark.parametrize(
    "masks",
    (
        torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
        torch.tensor([[1.0, 0.75, 0.25, 0.0], [0.0, 0.25, 0.75, 1.0]]),
        torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    ),
    ids=("hard", "soft", "overlap", "uncovered"),
)
def test_reference_normalization_equals_candidate_explicit_base_weights(
    masks: torch.Tensor,
) -> None:
    """Match base and regional normalized weights over every declared geometry."""

    candidate = StandardUnetAttentionWeightingPolicy().weights(
        masks,
        region_strengths=(
            REGIONAL_OWNERSHIP_STRENGTH,
            REGIONAL_OWNERSHIP_STRENGTH,
        ),
    )
    base = reference_base_mask(masks)
    weighted_regions = masks * REGIONAL_OWNERSHIP_STRENGTH
    denominator = (base + weighted_regions.sum(dim=0)).clamp_min(1e-6)

    torch.testing.assert_close(base / denominator, candidate.normalized_base)
    torch.testing.assert_close(
        weighted_regions / denominator.unsqueeze(0),
        candidate.normalized_regions,
    )


def test_parity_policy_retains_global_under_full_coverage() -> None:
    """Keep the accepted full-canvas global contribution under both regions."""

    masks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    characterized = StandardUnetAttentionWeightingPolicy().weights(
        masks,
        region_strengths=(
            REGIONAL_OWNERSHIP_STRENGTH,
            REGIONAL_OWNERSHIP_STRENGTH,
        ),
    )

    torch.testing.assert_close(
        characterized.normalized_base,
        torch.full((2,), 0.6),
    )


def test_reference_base_mask_rejects_non_floating_masks() -> None:
    """Fail before silently changing authored mask arithmetic."""

    with pytest.raises(TypeError, match="floating"):
        reference_base_mask(torch.ones((2, 2), dtype=torch.int64))
