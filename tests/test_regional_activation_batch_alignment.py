# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify activation batches adapt existing regional alignment authorities."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.regional_lora.activation_batch_alignment import (
    RegionalActivationBatchAlignmentResolver,
)


def test_resolver_composes_real_cfg_latent_and_tiled_authorities() -> None:
    """Adapt exact chunk and view batches without recreating their ordering."""

    contexts = _contexts(
        latent_batch_size=2,
        branches=(
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
        ),
    )
    layout = SpatialBatchLayout(
        8,
        4,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 4, 4, 4, 4),
            SpatialView(SpatialViewKind.TILE, 4, 0, 4, 4, 4, 4),
        ),
        input_batch_size=4,
    )

    alignment = RegionalActivationBatchAlignmentResolver().resolve(
        contexts,
        spatial_layout=layout,
    )

    assert alignment.latent_batch_size == 2
    assert alignment.chunk_count == 2
    assert alignment.base_batch_size == 4
    assert alignment.invocation_batch_size == 8
    assert alignment.spatial_layout is layout


def test_resolver_collapses_view_repeated_context_chunks_to_logical_cfg() -> None:
    """Count repeated spatial views as invocations, not new CFG branches."""

    contexts = _contexts(
        latent_batch_size=2,
        branches=(
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
        ),
    )
    layout = SpatialBatchLayout(
        8,
        4,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 4, 4, 4, 4),
            SpatialView(SpatialViewKind.TILE, 4, 0, 4, 4, 4, 4),
        ),
        input_batch_size=4,
    )

    alignment = RegionalActivationBatchAlignmentResolver().resolve(
        contexts,
        spatial_layout=layout,
    )

    assert alignment.latent_batch_size == 2
    assert alignment.chunk_count == 2
    assert alignment.base_batch_size == 4
    assert alignment.invocation_batch_size == 8


def test_resolver_rejects_nonrepeating_view_chunk_branches() -> None:
    """Fail closed when expanded contexts do not preserve view-major CFG order."""

    contexts = _contexts(
        latent_batch_size=2,
        branches=(
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
            RegionalAttentionBranch.NEGATIVE,
            RegionalAttentionBranch.POSITIVE,
        ),
    )
    layout = SpatialBatchLayout(
        8,
        4,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 4, 4, 4, 4),
            SpatialView(SpatialViewKind.TILE, 4, 0, 4, 4, 4, 4),
        ),
        input_batch_size=4,
    )

    with pytest.raises(ValueError, match="view-major branch order"):
        RegionalActivationBatchAlignmentResolver().resolve(
            contexts,
            spatial_layout=layout,
        )


def test_resolver_rejects_layout_that_disagrees_with_context_batch() -> None:
    """Refuse a view layout built over a different CFG/latent base batch."""

    contexts = _contexts(latent_batch_size=2)
    layout = SpatialBatchLayout(
        4,
        4,
        (SpatialView(SpatialViewKind.FULL, 0, 0, 4, 4, 4, 4),),
        input_batch_size=2,
    )

    with pytest.raises(ValueError, match="expanded context batch"):
        RegionalActivationBatchAlignmentResolver().resolve(
            contexts,
            spatial_layout=layout,
        )


def _contexts(
    *,
    latent_batch_size: int,
    branches: tuple[RegionalAttentionBranch, ...] = (
        RegionalAttentionBranch.POSITIVE,
        RegionalAttentionBranch.NEGATIVE,
    ),
) -> BatchedRegionalAttentionContexts:
    """Return one actual two-branch chunk-major alignment value."""

    chunks = tuple(
        RegionalAttentionChunkBatch(
            chunk_index,
            branch,
            chunk_index * latent_batch_size,
            (chunk_index + 1) * latent_batch_size,
        )
        for chunk_index, branch in enumerate(branches)
    )
    return BatchedRegionalAttentionContexts(
        latent_batch_size,
        chunks,
        torch.zeros((latent_batch_size * len(chunks), 1, 2)),
        (),
    )
