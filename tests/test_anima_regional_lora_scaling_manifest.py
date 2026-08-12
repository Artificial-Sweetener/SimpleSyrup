"""Verify the complete immutable P10.1 scaling declaration."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.anima_regional_lora_performance.matrix_manifest import (
    PerformanceAdapterLayout,
    PerformanceAttentionBackend,
    PerformanceEqualityMode,
    default_scaling_manifest,
)


def test_scaling_manifest_declares_every_position_and_exact_gate() -> None:
    """Pin region, adapter, backend, trace, equality, and cache expectations."""

    manifest = default_scaling_manifest()

    assert len(manifest.profiles) == 14
    assert tuple(profile.adapter_count for profile in manifest.profiles[:4]) == (
        0,
        1,
        2,
        4,
    )
    assert tuple(profile.region_count for profile in manifest.profiles[4:8]) == (
        2,
        4,
        2,
        4,
    )
    assert manifest.profiles[9].adapter_layout is (
        PerformanceAdapterLayout.REPEATED_PER_REGION
    )
    assert manifest.profiles[9].work.prepared_cache_entries == 448
    assert manifest.profiles[9].work.deduplicated_target_uses == 1344
    assert manifest.profiles[9].equality_mode is PerformanceEqualityMode.BF16
    assert manifest.profiles[10].work.active_adapter_uses == 1
    assert manifest.profiles[10].equality_mode is PerformanceEqualityMode.EXACT
    assert manifest.profiles[11].work.prepared_cache_entries == 0
    assert manifest.profiles[11].equality_mode is PerformanceEqualityMode.BF16
    assert all(
        profile.attention_backend is PerformanceAttentionBackend.SAGE
        for profile in manifest.profiles[-2:]
    )
    assert sum(profile.capture_operator_trace for profile in manifest.profiles) == 9
    assert manifest.profiles[1].maximum_overhead_percent == 15.0
    assert manifest.profiles[3].maximum_overhead_percent == 35.0


def test_scaling_manifest_requires_three_repeats_and_complete_order() -> None:
    """Reject weakened repeat evidence or a missing declared position."""

    with pytest.raises(ValueError, match="three repeats"):
        default_scaling_manifest(repeats=2)

    manifest = default_scaling_manifest()
    with pytest.raises(ValueError, match="complete order"):
        replace(manifest, profiles=manifest.profiles[:-1])


def test_scaling_profile_rejects_adapter_layout_mismatch() -> None:
    """Prevent attention-only declarations from carrying adapter policy."""

    manifest = default_scaling_manifest()
    with pytest.raises(ValueError, match="layout must agree"):
        replace(
            manifest.profiles[0],
            adapter_layout=PerformanceAdapterLayout.DISTINCT_STACKED,
        )
