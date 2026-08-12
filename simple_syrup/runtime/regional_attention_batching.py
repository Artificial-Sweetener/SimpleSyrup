# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Align processed regional contexts to Comfy's chunk-major model batch."""

from __future__ import annotations

import torch
from comfy.utils import repeat_to_batch_size

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from ..domain.regional_attention_selection import (
    REGIONAL_ATTENTION_SELECTION_SERVICE,
    ActiveProcessedRegionalAttentionChunk,
    RegionalAttentionSelectionService,
)


class RegionalAttentionBatchingService:
    """Build exact CFG-chunk and latent-batch aligned context tensors."""

    def __init__(
        self,
        selection: RegionalAttentionSelectionService | None = None,
    ) -> None:
        """Retain the authoritative active-entry selection collaborator."""

        self._selection = selection or REGIONAL_ATTENTION_SELECTION_SERVICE

    def align(
        self,
        plan: ProcessedRegionalAttentionPlan,
        *,
        base_context: torch.Tensor,
        cond_or_uncond: object,
        conditioning_uuids: object,
        sigma: float,
        latent_batch_size: int,
    ) -> BatchedRegionalAttentionContexts:
        """Return chunk-major base and canonical region context banks."""

        if isinstance(latent_batch_size, bool) or not isinstance(
            latent_batch_size, int
        ):
            raise TypeError("Regional attention latent_batch_size must be an integer.")
        if latent_batch_size < 1:
            raise ValueError("Regional attention latent_batch_size must be positive.")
        selected = self._selection.select_chunks(
            plan,
            cond_or_uncond=cond_or_uncond,
            conditioning_uuids=conditioning_uuids,
            sigma=sigma,
        )
        if not selected:
            raise ValueError(
                "Regional attention batching requires at least one Comfy chunk."
            )
        self._preflight(plan)

        expected_batch = len(selected) * latent_batch_size
        if (
            not isinstance(base_context, torch.Tensor)
            or base_context.ndim != 3
            or int(base_context.shape[0]) != expected_batch
        ):
            raise ValueError(
                "Runtime base context must match selected chunks and latent batch."
            )
        authority = plan.positive.base_context.entries[0].cross_attention
        if base_context.shape[1:] != authority.shape[1:]:
            raise ValueError(
                "Runtime base context sequence shape must match the regional plan."
            )

        chunk_values: list[RegionalAttentionChunkBatch] = []
        for chunk in selected:
            batch_start = chunk.chunk_index * latent_batch_size
            batch_stop = batch_start + latent_batch_size
            chunk_values.append(
                RegionalAttentionChunkBatch(
                    chunk.chunk_index,
                    chunk.branch,
                    batch_start,
                    batch_stop,
                )
            )
        return BatchedRegionalAttentionContexts(
            latent_batch_size=latent_batch_size,
            chunks=tuple(chunk_values),
            base_context=base_context,
            regions=tuple(
                self._align_region(
                    region_index,
                    selected,
                    latent_batch_size=latent_batch_size,
                    device=base_context.device,
                    dtype=base_context.dtype,
                )
                for region_index in range(plan.mask_bank.region_count)
            ),
        )

    @staticmethod
    def _align_region(
        region_index: int,
        chunks: tuple[ActiveProcessedRegionalAttentionChunk, ...],
        *,
        latent_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BatchedRegionalAttentionRegion:
        """Align every active entry slot for one canonical region."""

        sources = tuple(chunk.regional_entries[region_index] for chunk in chunks)
        entry_count = max(1 if not source else len(source) for source in sources)
        entries: list[BatchedRegionalAttentionEntry] = []
        for entry_index in range(entry_count):
            context_parts: list[torch.Tensor] = []
            strengths: list[float] = []
            for chunk, source in zip(chunks, sources, strict=True):
                if source is None:
                    entry = chunk.base_entry
                    strength = 1.0 if entry_index == 0 else 0.0
                elif entry_index < len(source):
                    entry = source[entry_index]
                    strength = entry.strength
                else:
                    entry = chunk.base_entry
                    strength = 0.0
                context_parts.append(
                    repeat_to_batch_size(entry.cross_attention, latent_batch_size).to(
                        device=device,
                        dtype=dtype,
                    )
                )
                strengths.extend((strength,) * latent_batch_size)
            entries.append(
                BatchedRegionalAttentionEntry(
                    entry_index,
                    torch.cat(context_parts, dim=0),
                    tuple(strengths),
                )
            )
        return BatchedRegionalAttentionRegion(region_index, tuple(entries))

    @staticmethod
    def _preflight(plan: ProcessedRegionalAttentionPlan) -> None:
        """Require all possible branch contexts to share model sequence state."""

        entries = tuple(
            entry
            for branch in (plan.positive, plan.negative)
            for context in (branch.base_context, *branch.regional_contexts)
            for entry in context.entries
        )
        authority = entries[0].cross_attention
        for entry in entries[1:]:
            tensor = entry.cross_attention
            if tensor.shape[1:] != authority.shape[1:]:
                raise ValueError(
                    "Regional attention context sequence shapes must match."
                )
            if tensor.device != authority.device:
                raise ValueError("Regional attention context devices must match.")
            if tensor.dtype != authority.dtype:
                raise ValueError("Regional attention context dtypes must match.")


REGIONAL_ATTENTION_BATCHING_SERVICE = RegionalAttentionBatchingService()
