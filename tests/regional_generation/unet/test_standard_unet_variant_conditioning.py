# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact native conditioning selection for standard-UNet variants."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.runtime.regional_attention_execution_context import (
    RegionalAttentionExecutionContext,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_conditioning import (
    StandardUnetVariantConditioningResolver,
)


def test_uniform_regional_rows_preserve_exact_context_identity() -> None:
    """Avoid copies when every CFG row selects the same native entry."""

    contexts = _contexts(((1.0, 1.0),))
    session = RegionalAttentionExecutionContext()
    resolver = StandardUnetVariantConditioningResolver(session)

    with session.activate(contexts):
        assert resolver.global_context() is contexts.base_context
        assert resolver.regional_context(0) is contexts.regions[0].entries[0].context


def test_inactive_cfg_row_uses_the_aligned_global_context() -> None:
    """Select regional positive and global negative rows without interpolation."""

    contexts = _contexts(((1.0, 0.0),))
    session = RegionalAttentionExecutionContext()
    resolver = StandardUnetVariantConditioningResolver(session)

    with session.activate(contexts):
        selected = resolver.regional_context(0)

    assert torch.equal(selected[0], contexts.regions[0].entries[0].context[0])
    assert torch.equal(selected[1], contexts.base_context[1])


def test_fractional_entry_strength_fails_closed() -> None:
    """Reject conditioning that native single-context execution cannot preserve."""

    contexts = _contexts(((1.0, 0.5),))
    session = RegionalAttentionExecutionContext()
    resolver = StandardUnetVariantConditioningResolver(session)

    with (
        session.activate(contexts),
        pytest.raises(ValueError, match="binary regional entry strengths"),
    ):
        resolver.regional_context(0)


def test_competing_entries_fail_before_graph_selection() -> None:
    """Reject rows requiring an attention-output combination."""

    contexts = _contexts(((1.0, 1.0), (1.0, 0.0)))
    session = RegionalAttentionExecutionContext()
    resolver = StandardUnetVariantConditioningResolver(session)

    with (
        session.activate(contexts),
        pytest.raises(ValueError, match="at most one active entry"),
    ):
        resolver.regional_context(0)


def test_invalid_region_fails_closed() -> None:
    """Reject a topology/context mismatch before selecting a graph context."""

    contexts = _contexts(((1.0, 1.0),))
    session = RegionalAttentionExecutionContext()
    resolver = StandardUnetVariantConditioningResolver(session)

    with (
        session.activate(contexts),
        pytest.raises(ValueError, match="outside conditioning"),
    ):
        resolver.regional_context(1)


def _contexts(
    strengths: tuple[tuple[float, float], ...],
) -> BatchedRegionalAttentionContexts:
    """Return one two-row CFG batch with the requested regional entries."""

    base = torch.tensor([[[10.0]], [[20.0]]])
    entries = tuple(
        BatchedRegionalAttentionEntry(
            entry_index,
            torch.tensor([[[100.0 + entry_index]], [[200.0 + entry_index]]]),
            entry_strengths,
        )
        for entry_index, entry_strengths in enumerate(strengths)
    )
    return BatchedRegionalAttentionContexts(
        1,
        (
            RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.NEGATIVE, 1, 2),
        ),
        base,
        (BatchedRegionalAttentionRegion(0, entries),),
    )
