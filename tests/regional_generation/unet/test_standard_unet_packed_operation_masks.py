# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify regional LoRA ownership inside compact standard-UNet attn2 rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simple_syrup.domain.regional_activation_geometry import RegionalActivationLayout
from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)
from simple_syrup.runtime.regional_lora.standard_unet_packed_operation_masks import (
    StandardUnetPackedOperationMaskResolver,
)


@dataclass(frozen=True, slots=True)
class _Use:
    """Expose operation ownership for one synthetic target use."""

    composition_index: int
    region_index: int
    branch: RegionalLoraBranch


def test_packed_image_masks_exclude_base_and_preserve_cfg_ownership() -> None:
    """Apply spatial masks only to matching regional query and output rows."""

    execution = _execution()
    masks = StandardUnetPackedOperationMaskResolver().resolve_image_tokens(
        execution,
        uses=(
            _Use(0, 0, RegionalLoraBranch.POSITIVE),
            _Use(1, 0, RegionalLoraBranch.NEGATIVE),
        ),
        inputs=torch.zeros((4, 2, 8)),
    )

    assert masks.geometry.layout is RegionalActivationLayout.CONSUMER_SPATIALIZED
    assert masks.multipliers.shape == (2, 4, 2, 1)
    torch.testing.assert_close(
        masks.multipliers[0, :, :, 0],
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.5], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        masks.multipliers[1, :, :, 0],
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.25, 1.0]]),
    )


def test_packed_context_masks_gate_rows_without_reshaping_tokens() -> None:
    """Broadcast branch ownership over the untouched context-token sequence."""

    execution = _execution()
    masks = StandardUnetPackedOperationMaskResolver().resolve_context_tokens(
        execution,
        uses=(
            _Use(0, 0, RegionalLoraBranch.POSITIVE),
            _Use(1, 0, RegionalLoraBranch.NEGATIVE),
        ),
        inputs=torch.zeros((4, 77, 64)),
    )

    assert masks.geometry.layout is RegionalActivationLayout.BRANCH_TOKENS
    assert masks.geometry.invocation_shape == (4, 77, 64)
    assert masks.multipliers.shape == (2, 4, 77, 1)
    assert masks.multipliers[0, :2].eq(0.0).all()
    assert masks.multipliers[0, 2].eq(1.0).all()
    assert masks.multipliers[0, 3].eq(0.0).all()
    assert masks.multipliers[1, 3].eq(1.0).all()


def _execution() -> UnetAttn2Execution:
    """Return two CFG rows and one compact regional branch over both rows."""

    base = torch.zeros((2, 77, 64))
    contexts = BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=(
            RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.NEGATIVE, 1, 2),
        ),
        base_context=base,
        regions=(
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, base.clone(), (1.0, 1.0)),),
            ),
        ),
    )
    return UnetAttn2Execution(
        contexts,
        torch.tensor([[[1.0, 0.5], [0.25, 1.0]]]),
        (0.75,),
        1,
        2,
    )
