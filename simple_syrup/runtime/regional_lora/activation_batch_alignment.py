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
        return RegionalActivationBatchAlignment(
            latent_batch_size=contexts.latent_batch_size,
            chunk_count=len(contexts.chunks),
            spatial_layout=spatial_layout,
        )


REGIONAL_ACTIVATION_BATCH_ALIGNMENT_RESOLVER = (
    RegionalActivationBatchAlignmentResolver()
)
