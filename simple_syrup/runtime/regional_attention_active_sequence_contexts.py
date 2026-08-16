# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve the context tensors participating in one regional model call."""

from __future__ import annotations

import torch

from ..domain.regional_attention_selection import (
    ActiveProcessedRegionalAttentionChunk,
)


class RegionalAttentionActiveSequenceContextResolver:
    """Own schedule-scoped sequence-alignment membership."""

    @staticmethod
    def resolve(
        base_context: torch.Tensor,
        chunks: tuple[ActiveProcessedRegionalAttentionChunk, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return the runtime base followed by active regional contexts."""

        if not isinstance(base_context, torch.Tensor) or base_context.ndim != 3:
            raise TypeError(
                "Active regional sequence resolution requires a BxSxD base tensor."
            )
        if not isinstance(chunks, tuple) or not chunks:
            raise ValueError(
                "Active regional sequence resolution requires selected chunks."
            )
        if any(
            not isinstance(chunk, ActiveProcessedRegionalAttentionChunk)
            for chunk in chunks
        ):
            raise TypeError(
                "Active regional sequence resolution received an invalid chunk."
            )
        return (
            base_context,
            *(
                entry.cross_attention
                for chunk in chunks
                for entries in chunk.regional_entries
                if entries
                for entry in entries
            ),
        )


REGIONAL_ATTENTION_ACTIVE_SEQUENCE_CONTEXT_RESOLVER = (
    RegionalAttentionActiveSequenceContextResolver()
)
