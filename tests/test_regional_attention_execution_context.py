"""Verify model-neutral regional attention context lifetime ownership."""

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


def test_execution_context_publishes_nested_values_and_restores_each_scope() -> None:
    """Restore outer and empty state after exact nested model-call scopes."""

    owner = RegionalAttentionExecutionContext()
    outer = _contexts(1.0)
    inner = _contexts(2.0)

    with owner.activate(outer):
        assert owner.require_current() is outer
        with owner.activate(inner):
            assert owner.require_current() is inner
        assert owner.require_current() is outer

    with pytest.raises(RuntimeError, match="outside"):
        owner.require_current()


def test_execution_context_restores_state_after_nested_failure() -> None:
    """Prevent failed model calls from leaking active context state."""

    owner = RegionalAttentionExecutionContext()
    contexts = _contexts(1.0)

    with pytest.raises(RuntimeError, match="sentinel"):
        with owner.activate(contexts):
            raise RuntimeError("sentinel")

    with pytest.raises(RuntimeError, match="outside"):
        owner.require_current()


def test_execution_context_returns_validated_fallback_only_when_inactive() -> None:
    """Preserve the static harness contract without overriding active state."""

    owner = RegionalAttentionExecutionContext()
    fallback = _contexts(1.0)
    active = _contexts(2.0)

    assert owner.current_or(fallback) is fallback
    with owner.activate(active):
        assert owner.current_or(fallback) is active

    with pytest.raises(TypeError, match="fallback"):
        owner.current_or(object())  # type: ignore[arg-type]


def _contexts(value: float) -> BatchedRegionalAttentionContexts:
    """Return one minimal aligned context batch with recognizable identity."""

    context = torch.full((1, 1, 1), value)
    return BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=(
            RegionalAttentionChunkBatch(
                0,
                RegionalAttentionBranch.POSITIVE,
                0,
                1,
            ),
        ),
        base_context=context,
        regions=(
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, context, (1.0,)),),
            ),
        ),
    )
