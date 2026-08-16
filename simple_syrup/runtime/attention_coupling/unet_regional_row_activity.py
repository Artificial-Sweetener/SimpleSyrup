# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve authored active regional rows for standard-UNet attention."""

from __future__ import annotations

import torch

from ...domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionPlan,
)
from ...domain.regional_attention import RegionalAttentionBranch
from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts


class StandardUnetRegionalRowActivityResolver:
    """Own reference-faithful regional support across CFG and latent rows."""

    def resolve(
        self,
        plan: ProcessedRegionalAttentionPlan,
        contexts: BatchedRegionalAttentionContexts,
    ) -> torch.Tensor:
        """Return one boolean R/B matrix for authored active regional rows."""

        if not isinstance(plan, ProcessedRegionalAttentionPlan):
            raise TypeError("Standard UNet row activity requires a processed plan.")
        if not isinstance(contexts, BatchedRegionalAttentionContexts):
            raise TypeError("Standard UNet row activity requires aligned contexts.")
        if len(contexts.regions) != plan.mask_bank.region_count:
            raise ValueError(
                "Standard UNet row activity region count must match the mask bank."
            )
        activity = torch.zeros(
            (plan.mask_bank.region_count, int(contexts.base_context.shape[0])),
            dtype=torch.bool,
            device=contexts.base_context.device,
        )
        for chunk in contexts.chunks:
            branch = self._branch(plan, chunk.branch)
            for region in contexts.regions:
                if region.region_index >= len(branch.regional_contexts):
                    continue
                row_activity = torch.zeros(
                    chunk.batch_stop - chunk.batch_start,
                    dtype=torch.bool,
                    device=activity.device,
                )
                for entry in region.entries:
                    strengths = activity.new_tensor(
                        entry.strengths[chunk.batch_start : chunk.batch_stop],
                        dtype=contexts.base_context.dtype,
                    )
                    row_activity |= strengths.ne(0)
                activity[
                    region.region_index,
                    chunk.batch_start : chunk.batch_stop,
                ] = row_activity
        return activity

    @staticmethod
    def _branch(
        plan: ProcessedRegionalAttentionPlan,
        branch: RegionalAttentionBranch,
    ) -> ProcessedRegionalAttentionBranch:
        """Return the exact processed branch selected by one model chunk."""

        if branch is RegionalAttentionBranch.POSITIVE:
            return plan.positive
        if branch is RegionalAttentionBranch.NEGATIVE:
            return plan.negative
        raise ValueError(f"Unsupported regional attention branch: {branch!r}")


STANDARD_UNET_REGIONAL_ROW_ACTIVITY_RESOLVER = StandardUnetRegionalRowActivityResolver()
