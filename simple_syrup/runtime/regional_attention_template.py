# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build a validation-only static context batch for dynamic attention patches."""

from __future__ import annotations

import torch
from comfy.utils import repeat_to_batch_size

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..domain.regional_attention import RegionalAttentionBranch
from ..domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)


def build_regional_attention_template(
    plan: ProcessedRegionalAttentionPlan,
    *,
    latent_batch_size: int,
) -> BatchedRegionalAttentionContexts:
    """Return one positive-branch fallback used only outside dynamic call scope."""

    if not isinstance(plan, ProcessedRegionalAttentionPlan):
        raise TypeError("Regional attention template requires a processed plan.")
    if isinstance(latent_batch_size, bool) or not isinstance(latent_batch_size, int):
        raise TypeError("Regional attention template batch size must be an integer.")
    if latent_batch_size < 1:
        raise ValueError("Regional attention template batch size must be positive.")

    base_entry = plan.positive.base_context.entries[0]
    base_context = repeat_to_batch_size(
        base_entry.cross_attention,
        latent_batch_size,
    )
    regions = tuple(
        _template_region(
            region_index,
            plan=plan,
            base_context=base_context,
            latent_batch_size=latent_batch_size,
        )
        for region_index in range(plan.mask_bank.region_count)
    )
    return BatchedRegionalAttentionContexts(
        latent_batch_size=latent_batch_size,
        chunks=(
            RegionalAttentionChunkBatch(
                0,
                RegionalAttentionBranch.POSITIVE,
                0,
                latent_batch_size,
            ),
        ),
        base_context=base_context,
        regions=regions,
    )


def _template_region(
    region_index: int,
    *,
    plan: ProcessedRegionalAttentionPlan,
    base_context: torch.Tensor,
    latent_batch_size: int,
) -> BatchedRegionalAttentionRegion:
    """Return one representative region while preserving canonical structure."""

    if region_index < len(plan.positive.regional_contexts):
        entry = plan.positive.regional_contexts[region_index].entries[0]
        strength = entry.strength
    else:
        entry = plan.positive.base_context.entries[0]
        strength = 1.0
    aligned = repeat_to_batch_size(entry.cross_attention, latent_batch_size)
    if aligned.shape[1:] != base_context.shape[1:]:
        raise ValueError("Regional attention template sequence shapes must match.")
    return BatchedRegionalAttentionRegion(
        region_index,
        (
            BatchedRegionalAttentionEntry(
                0,
                aligned,
                (strength,) * latent_batch_size,
            ),
        ),
    )
