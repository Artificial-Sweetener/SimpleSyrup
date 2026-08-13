# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt existing CFG, latent-batch, and spatial-view alignment authorities."""

from __future__ import annotations

from ...domain.regional_activation_geometry import RegionalActivationBatchAlignment
from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...domain.spatial_views import SpatialBatchLayout


class RegionalActivationBatchAlignmentResolver:
    """Compose published alignment values without recreating their policies."""

    def resolve(
        self,
        contexts: BatchedRegionalAttentionContexts,
        *,
        spatial_layout: SpatialBatchLayout | None = None,
    ) -> RegionalActivationBatchAlignment:
        """Return one activation alignment from exact existing authorities."""

        if not isinstance(contexts, BatchedRegionalAttentionContexts):
            raise TypeError(
                "Regional activation alignment requires batched regional contexts."
            )
        chunk_count = len(contexts.chunks)
        if spatial_layout is not None:
            chunk_count = self._logical_chunk_count(contexts, spatial_layout)
        return RegionalActivationBatchAlignment(
            latent_batch_size=contexts.latent_batch_size,
            chunk_count=chunk_count,
            spatial_layout=spatial_layout,
        )

    @staticmethod
    def _logical_chunk_count(
        contexts: BatchedRegionalAttentionContexts,
        layout: SpatialBatchLayout,
    ) -> int:
        """Collapse exact view-major repetitions to their logical CFG chunks."""

        if not isinstance(layout, SpatialBatchLayout):
            raise TypeError(
                "Regional activation spatial layout must be a SpatialBatchLayout."
            )
        latent_batch_size = contexts.latent_batch_size
        if layout.input_batch_size % latent_batch_size != 0:
            raise ValueError(
                "Regional activation spatial layout input batch must divide "
                "evenly across the latent batch."
            )
        context_batch_size = int(contexts.base_context.shape[0])
        if context_batch_size != layout.expanded_batch_size:
            raise ValueError(
                "Regional activation expanded context batch must match the "
                "published spatial layout."
            )
        logical_chunk_count = layout.input_batch_size // latent_batch_size
        expected_chunk_count = logical_chunk_count * layout.view_count
        if len(contexts.chunks) != expected_chunk_count:
            raise ValueError(
                "Regional activation context chunks must match the published "
                "view and input batches."
            )
        branches = tuple(chunk.branch for chunk in contexts.chunks)
        logical_branches = branches[:logical_chunk_count]
        if branches != logical_branches * layout.view_count:
            raise ValueError(
                "Regional activation context chunks must retain view-major "
                "branch order."
            )
        return logical_chunk_count


REGIONAL_ACTIVATION_BATCH_ALIGNMENT_RESOLVER = (
    RegionalActivationBatchAlignmentResolver()
)
