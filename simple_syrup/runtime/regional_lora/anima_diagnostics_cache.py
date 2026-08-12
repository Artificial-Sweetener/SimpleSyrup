# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the bounded immutable Anima diagnostics snapshot cache."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from ...domain.spatial_views import SpatialBatchLayout
from ..regional_attention_model_call_values import RegionalAttentionModelCallValues
from ..regional_lora_schedule_resolution import RegionalLoraScheduleResolution
from .anima_activation_context import AnimaActivationGeometry
from .anima_diagnostic_values import AnimaRegionalExecutionDiagnostics

AnimaDiagnosticsCacheKey = tuple[
    AnimaActivationGeometry,
    SpatialBatchLayout,
    RegionalLoraScheduleResolution,
    int,
    int,
    RegionalAttentionModelCallValues,
]


class AnimaDiagnosticsSnapshotCache:
    """Retain a bounded thread-safe LRU of immutable final snapshots."""

    def __init__(self, maximum_entries: int = 32) -> None:
        """Create one cache with a positive explicit capacity."""

        if isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int):
            raise TypeError("Anima diagnostics cache capacity must be an integer.")
        if maximum_entries < 1:
            raise ValueError("Anima diagnostics cache capacity must be positive.")
        self._maximum_entries = maximum_entries
        self._entries: OrderedDict[
            AnimaDiagnosticsCacheKey,
            AnimaRegionalExecutionDiagnostics,
        ] = OrderedDict()
        self._lock = Lock()

    def get(
        self,
        key: AnimaDiagnosticsCacheKey,
    ) -> AnimaRegionalExecutionDiagnostics | None:
        """Return and refresh one cached snapshot when present."""

        with self._lock:
            snapshot = self._entries.get(key)
            if snapshot is not None:
                self._entries.move_to_end(key)
            return snapshot

    def store(
        self,
        key: AnimaDiagnosticsCacheKey,
        snapshot: AnimaRegionalExecutionDiagnostics,
    ) -> AnimaRegionalExecutionDiagnostics:
        """Store one immutable snapshot and evict the least-recently-used value."""

        if not isinstance(snapshot, AnimaRegionalExecutionDiagnostics):
            raise TypeError("Anima diagnostics cache requires an immutable snapshot.")
        with self._lock:
            self._entries[key] = snapshot
            self._entries.move_to_end(key)
            if len(self._entries) > self._maximum_entries:
                self._entries.popitem(last=False)
        return snapshot
