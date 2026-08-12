"""Verify bounded immutable Anima diagnostic snapshot caching."""

from __future__ import annotations

import pytest

from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.regional_attention_diagnostic_values import (
    RegionalAttentionChunkDiagnostics,
    RegionalAttentionViewDiagnostics,
)
from simple_syrup.runtime.regional_attention_model_call_values import (
    RegionalAttentionModelCallValues,
)
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_diagnostic_values import (
    AnimaRegionalExecutionDiagnostics,
    AnimaWorkEstimateDiagnostics,
)
from simple_syrup.runtime.regional_lora.anima_diagnostics_cache import (
    AnimaDiagnosticsSnapshotCache,
)
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)


def test_anima_diagnostics_cache_is_bounded_and_least_recently_used() -> None:
    """Evict the oldest unrefreshed immutable snapshot at capacity."""

    cache = AnimaDiagnosticsSnapshotCache(maximum_entries=2)
    first = _snapshot("first")
    second = _snapshot("second")
    third = _snapshot("third")

    assert cache.store(_key(0), first) is first
    assert cache.store(_key(1), second) is second
    assert cache.get(_key(0)) is first
    assert cache.store(_key(2), third) is third

    assert cache.get(_key(0)) is first
    assert cache.get(_key(1)) is None
    assert cache.get(_key(2)) is third


@pytest.mark.parametrize("capacity", [0, -1])
def test_anima_diagnostics_cache_rejects_nonpositive_capacity(capacity: int) -> None:
    """Reject an unbounded or unusable cache configuration."""

    with pytest.raises(ValueError, match="capacity must be positive"):
        AnimaDiagnosticsSnapshotCache(capacity)


def test_anima_diagnostics_cache_rejects_boolean_capacity() -> None:
    """Reject boolean values at the integer cache boundary."""

    with pytest.raises(TypeError, match="capacity must be an integer"):
        AnimaDiagnosticsSnapshotCache(True)


def _key(
    prepared_cache_entries: int,
) -> tuple[
    AnimaActivationGeometry,
    SpatialBatchLayout,
    RegionalLoraScheduleResolution,
    int,
    int,
    RegionalAttentionModelCallValues,
]:
    """Return one structurally valid immutable diagnostics cache key."""

    return (
        AnimaActivationGeometry(1, 1, 1, 1, 1, 1, 1, 1, 1, None),
        _layout(),
        RegionalLoraScheduleResolution((), ()),
        prepared_cache_entries,
        100 + prepared_cache_entries,
        RegionalAttentionModelCallValues(float(prepared_cache_entries), ()),
    )


def _layout() -> SpatialBatchLayout:
    """Return one exact full-canvas layout."""

    return SpatialBatchLayout(
        1,
        1,
        (SpatialView(SpatialViewKind.FULL, 0, 0, 1, 1, 1, 1),),
        1,
    )


def _snapshot(backend: str) -> AnimaRegionalExecutionDiagnostics:
    """Return one minimal immutable final snapshot."""

    return AnimaRegionalExecutionDiagnostics(
        strategy="attention_coupling",
        backend=backend,
        region_count=0,
        coverage_class="all_base",
        active_region_indices=(),
        canonical_width=1,
        canonical_height=1,
        region_coverage=(),
        uncovered_fraction=1.0,
        overlap_fraction=0.0,
        spatial_mode="full",
        views=(RegionalAttentionViewDiagnostics(0, "full", 0, 0, 1, 1, 1, 1),),
        query_time=1,
        query_height=1,
        query_width=1,
        query_token_count=1,
        input_batch_size=1,
        latent_batch_size=1,
        layout_input_batch_size=1,
        expanded_view_batch_size=1,
        active_branches=("positive",),
        positive_chunk_count=1,
        negative_chunk_count=0,
        chunks=(RegionalAttentionChunkDiagnostics(0, "positive", 0, 1),),
        sampling_sigma=1.0,
        conditioning_uuids=(),
        regional_entries=(),
        adapter_uses=(),
        prepared_cache_entries=0,
        work=AnimaWorkEstimateDiagnostics(1.0, 0.0, 0, 0, 0, 0, 0, 0, 1.0),
    )
