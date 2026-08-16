# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Align processed regional contexts to Comfy's chunk-major model batch."""

from __future__ import annotations

import torch
from comfy.utils import repeat_to_batch_size

from ..domain.processed_regional_attention import (
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
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
from .regional_attention_active_sequence_contexts import (
    REGIONAL_ATTENTION_ACTIVE_SEQUENCE_CONTEXT_RESOLVER,
    RegionalAttentionActiveSequenceContextResolver,
)
from .regional_attention_sequence_alignment import (
    RegionalAttentionSequenceAligner,
)


class RegionalAttentionBatchingService:
    """Build exact CFG-chunk and latent-batch aligned context tensors."""

    def __init__(
        self,
        selection: RegionalAttentionSelectionService | None = None,
        sequence_aligner: RegionalAttentionSequenceAligner | None = None,
        active_sequence_contexts: RegionalAttentionActiveSequenceContextResolver = (
            REGIONAL_ATTENTION_ACTIVE_SEQUENCE_CONTEXT_RESOLVER
        ),
    ) -> None:
        """Retain authoritative selection and sequence-alignment collaborators."""

        self._selection = selection or REGIONAL_ATTENTION_SELECTION_SERVICE
        if sequence_aligner is not None and not isinstance(
            sequence_aligner, RegionalAttentionSequenceAligner
        ):
            raise TypeError("Regional batching requires a sequence aligner.")
        self._sequence_aligner = sequence_aligner
        if not isinstance(
            active_sequence_contexts,
            RegionalAttentionActiveSequenceContextResolver,
        ):
            raise TypeError(
                "Regional batching requires an active sequence context resolver."
            )
        self._active_sequence_contexts = active_sequence_contexts

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
        if int(base_context.shape[2]) != int(authority.shape[2]):
            raise ValueError(
                "Runtime base context feature width must match the regional plan."
            )
        target_sequence_length = int(base_context.shape[1])
        aligned_base_context = base_context
        if self._sequence_aligner is not None:
            target_sequence_length = self._sequence_aligner.target_length(
                self._active_sequence_contexts.resolve(base_context, selected)
            )
            aligned_base_context = self._sequence_aligner.align(
                base_context,
                target_length=target_sequence_length,
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
            base_context=aligned_base_context,
            regions=tuple(
                self._align_region(
                    region_index,
                    selected,
                    latent_batch_size=latent_batch_size,
                    device=aligned_base_context.device,
                    dtype=aligned_base_context.dtype,
                    target_sequence_length=target_sequence_length,
                )
                for region_index in range(plan.mask_bank.region_count)
            ),
        )

    def _align_region(
        self,
        region_index: int,
        chunks: tuple[ActiveProcessedRegionalAttentionChunk, ...],
        *,
        latent_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        target_sequence_length: int,
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
                repeated = repeat_to_batch_size(
                    entry.cross_attention,
                    latent_batch_size,
                ).to(
                    device=device,
                    dtype=dtype,
                )
                if self._sequence_aligner is not None:
                    repeated = self._sequence_aligner.align(
                        repeated,
                        target_length=target_sequence_length,
                    )
                context_parts.append(repeated)
                strengths.extend((strength,) * latent_batch_size)
            entries.append(
                BatchedRegionalAttentionEntry(
                    entry_index,
                    torch.cat(context_parts, dim=0),
                    tuple(strengths),
                )
            )
        return BatchedRegionalAttentionRegion(region_index, tuple(entries))

    def _preflight(self, plan: ProcessedRegionalAttentionPlan) -> None:
        """Require all possible branch contexts to share execution state."""

        entries = self._entries(plan)
        authority = entries[0].cross_attention
        for entry in entries[1:]:
            tensor = entry.cross_attention
            shape_mismatch = (
                int(tensor.shape[2]) != int(authority.shape[2])
                if self._sequence_aligner is not None
                else tensor.shape[1:] != authority.shape[1:]
            )
            if shape_mismatch:
                raise ValueError(
                    "Regional attention context execution shapes must match."
                )
            if tensor.device != authority.device:
                raise ValueError("Regional attention context devices must match.")
            if tensor.dtype != authority.dtype:
                raise ValueError("Regional attention context dtypes must match.")

    @staticmethod
    def _entries(
        plan: ProcessedRegionalAttentionPlan,
    ) -> tuple[ProcessedRegionalAttentionEntry, ...]:
        """Return every processed entry in stable branch and region order."""

        return tuple(
            entry
            for branch in (plan.positive, plan.negative)
            for context in (branch.base_context, *branch.regional_contexts)
            for entry in context.entries
        )


REGIONAL_ATTENTION_BATCHING_SERVICE = RegionalAttentionBatchingService()
