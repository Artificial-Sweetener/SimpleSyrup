# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify target-use masks compose spatial and CFG-branch ownership exactly."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from simple_syrup.domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationGeometry,
    RegionalActivationLayout,
)
from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.masking.regional_activation_mask_projection import (
    RegionalActivationMaskBatch,
)
from simple_syrup.runtime.regional_lora.operation_mask_resolution import (
    RegionalOperationMaskResolver,
)


@dataclass(frozen=True, slots=True)
class _Use:
    """Expose the minimal immutable target-use ownership contract."""

    composition_index: int
    region_index: int
    branch: RegionalLoraBranch


def test_resolver_combines_region_and_branch_in_target_use_order() -> None:
    """Gate every projected region to its authored positive or negative rows."""

    spatial = _spatial_masks(
        torch.tensor(
            [
                [[1.0, 0.5], [1.0, 0.5]],
                [[0.25, 0.75], [0.25, 0.75]],
            ]
        )
    )
    contexts = _contexts(
        latent_batch_size=1,
        branches=(
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
        ),
    )
    uses = (
        _Use(2, 1, RegionalLoraBranch.NEGATIVE),
        _Use(5, 0, RegionalLoraBranch.POSITIVE),
    )

    resolved = RegionalOperationMaskResolver().resolve(
        spatial,
        contexts=contexts,
        uses=uses,
    )

    assert resolved.composition_indices == (2, 5)
    torch.testing.assert_close(
        resolved.multipliers.squeeze(-1),
        torch.tensor(
            [
                [[0.0, 0.0], [0.25, 0.75]],
                [[1.0, 0.5], [0.0, 0.0]],
            ]
        ),
    )


def test_resolver_retains_view_major_branch_repetition() -> None:
    """Repeat the logical CFG gate exactly once for every published view."""

    layout = SpatialBatchLayout(
        2,
        1,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 1, 1, 1, 1),
            SpatialView(SpatialViewKind.TILE, 1, 0, 1, 1, 1, 1),
        ),
        input_batch_size=2,
    )
    spatial = _spatial_masks(
        torch.ones((1, 4, 1)),
        alignment=RegionalActivationBatchAlignment(1, 2, layout),
    )
    contexts = _contexts(
        latent_batch_size=1,
        branches=(
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
        ),
    )

    resolved = RegionalOperationMaskResolver().resolve(
        spatial,
        contexts=contexts,
        uses=(_Use(0, 0, RegionalLoraBranch.POSITIVE),),
    )

    assert resolved.multipliers[:, :, 0, 0].tolist() == [[1.0, 0.0, 1.0, 0.0]]


def test_resolver_rejects_noncanonical_use_composition_order() -> None:
    """Require target uses to preserve global adapter composition order."""

    with pytest.raises(ValueError, match="composition order"):
        RegionalOperationMaskResolver().resolve(
            _spatial_masks(torch.ones((1, 2, 2))),
            contexts=_contexts(
                latent_batch_size=1,
                branches=(
                    RegionalAttentionBranch.POSITIVE,
                    RegionalAttentionBranch.NEGATIVE,
                ),
            ),
            uses=(
                _Use(3, 0, RegionalLoraBranch.POSITIVE),
                _Use(1, 0, RegionalLoraBranch.NEGATIVE),
            ),
        )


def test_resolver_rejects_unavailable_region_before_execution() -> None:
    """Fail before operation math when a target use lacks a projected region."""

    with pytest.raises(ValueError, match="unavailable region"):
        RegionalOperationMaskResolver().resolve(
            _spatial_masks(torch.ones((1, 2, 2))),
            contexts=_contexts(
                latent_batch_size=1,
                branches=(
                    RegionalAttentionBranch.POSITIVE,
                    RegionalAttentionBranch.NEGATIVE,
                ),
            ),
            uses=(_Use(0, 1, RegionalLoraBranch.POSITIVE),),
        )


def _spatial_masks(
    values: torch.Tensor,
    *,
    alignment: RegionalActivationBatchAlignment | None = None,
) -> RegionalActivationMaskBatch:
    """Wrap R/B/S masks in explicit consumer-spatialized geometry."""

    _, batch, tokens = values.shape
    resolved_alignment = alignment or RegionalActivationBatchAlignment(batch, 1)
    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.CONSUMER_SPATIALIZED,
        (batch, tokens, 3),
        2,
        1,
        tokens,
        resolved_alignment,
    )
    return RegionalActivationMaskBatch(values.unsqueeze(-1), geometry)


def _contexts(
    *,
    latent_batch_size: int,
    branches: tuple[RegionalAttentionBranch, ...],
) -> BatchedRegionalAttentionContexts:
    """Return exact chunk-major contexts for the supplied branch sequence."""

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
        torch.zeros((len(chunks) * latent_batch_size, 1, 2)),
        (),
    )
