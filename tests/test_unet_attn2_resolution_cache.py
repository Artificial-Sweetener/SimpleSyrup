"""Verify task-local standard-UNet attn2 resolution caching."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_resolution_cache import (
    StandardUnetAttn2ResolutionCache,
    StandardUnetAttn2ResolutionKey,
)


def test_resolution_cache_reuses_one_execution_per_call_local_key() -> None:
    """Build one exact resolution once while its diffusion scope is active."""

    cache = StandardUnetAttn2ResolutionCache()
    key = _key(_layout())
    calls = 0

    def factory() -> UnetAttn2Execution:
        """Return a new execution while counting real cache misses."""

        nonlocal calls
        calls += 1
        return _execution()

    with cache.activate():
        first = cache.resolve(key, factory)
        second = cache.resolve(key, factory)

        assert second is first
        assert calls == 1
        assert cache.size == 1

    with pytest.raises(RuntimeError, match="outside"):
        cache.resolve(key, factory)


def test_resolution_cache_nesting_and_failure_restore_outer_state() -> None:
    """Isolate nested calls and clear failed inner scopes exactly."""

    cache = StandardUnetAttn2ResolutionCache()
    key = _key(_layout())
    outer = _execution()
    inner = _execution()

    with cache.activate():
        assert cache.resolve(key, lambda: outer) is outer
        with pytest.raises(RuntimeError, match="nested failure"):
            with cache.activate():
                assert cache.resolve(key, lambda: inner) is inner
                raise RuntimeError("nested failure")
        assert cache.resolve(key, lambda: inner) is outer
        assert cache.size == 1

    with pytest.raises(RuntimeError, match="outside"):
        _ = cache.size


def test_resolution_key_requires_exact_layout_identity_and_every_resolution_axis() -> (
    None
):
    """Separate equal foreign layouts and every projection-changing dimension."""

    layout = _layout()
    key = _key(layout)

    assert key == _key(layout)
    assert key != _key(_layout())
    assert key != replace(key, input_batch_size=2)
    assert key != replace(key, query_height=1)
    assert key != replace(key, query_width=8)
    assert key != replace(key, original_height=16)
    assert key != replace(key, original_width=16)
    assert key != replace(key, dtype=torch.float64)


def test_resolution_factory_failure_does_not_poison_cache() -> None:
    """Retry a failed build instead of retaining partial resolution state."""

    cache = StandardUnetAttn2ResolutionCache()
    key = _key(_layout())
    attempts = 0

    def factory() -> UnetAttn2Execution:
        """Fail once, then return a complete execution."""

        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("projection failed")
        return _execution()

    with cache.activate():
        with pytest.raises(RuntimeError, match="projection failed"):
            cache.resolve(key, factory)
        assert cache.size == 0
        result = cache.resolve(key, factory)

        assert isinstance(result, UnetAttn2Execution)
        assert attempts == 2
        assert cache.size == 1


def _key(layout: SpatialBatchLayout) -> StandardUnetAttn2ResolutionKey:
    """Return one exact CPU float32 rectangular-resolution key."""

    return StandardUnetAttn2ResolutionKey(
        input_batch_size=1,
        query_height=2,
        query_width=4,
        original_height=4,
        original_width=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        published_layout=layout,
    )


def _layout() -> SpatialBatchLayout:
    """Return one explicit tile layout."""

    return SpatialBatchLayout(
        8,
        4,
        (SpatialView(SpatialViewKind.TILE, 0, 0, 8, 4, 8, 4),),
        1,
    )


def _execution() -> UnetAttn2Execution:
    """Return one minimal preprojected execution."""

    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        torch.ones(1, 1, 1),
        (
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, torch.ones(1, 1, 1), (1.0,)),),
            ),
        ),
    )
    return UnetAttn2Execution(contexts, torch.ones(1, 1, 8), (1.0,))
