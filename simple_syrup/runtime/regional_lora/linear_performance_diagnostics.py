# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Aggregate standard-UNet regional Linear route diagnostics."""

from __future__ import annotations

import logging
from collections import Counter
from threading import Lock

LOGGER = logging.getLogger(__name__)


class RegionalLinearPerformanceDiagnostics:
    """Collect bounded route counts and emit one lifecycle summary."""

    def __init__(self) -> None:
        """Create empty thread-safe aggregate counters."""

        self._lock = Lock()
        self._partitioned_calls = 0
        self._mapped_calls = 0
        self._declines: Counter[str] = Counter()
        self._group_counts: Counter[int] = Counter()
        self._support_counts: Counter[tuple[int, ...]] = Counter()

    def record_partitioned(
        self,
        active_groups: tuple[int, ...],
        support_counts: tuple[int, ...],
    ) -> None:
        """Record one base-once projection without logging its target."""

        with self._lock:
            self._partitioned_calls += 1
            self._group_counts[len(active_groups)] += 1
            self._support_counts[support_counts] += 1

    def record_decline(self, reason: str) -> None:
        """Record one bounded fallback reason."""

        with self._lock:
            self._declines[reason] += 1

    def record_mapped(self, active_groups: tuple[int, ...]) -> None:
        """Record one compatible unique-target fused projection."""

        with self._lock:
            self._mapped_calls += 1
            self._group_counts[len(active_groups)] += 1

    def emit_and_clear(self) -> None:
        """Log one aggregate summary and release all retained counters."""

        with self._lock:
            partitioned_calls = self._partitioned_calls
            mapped_calls = self._mapped_calls
            declines = dict(self._declines)
            group_counts = dict(self._group_counts)
            most_common_support = self._support_counts.most_common(12)
            self._partitioned_calls = 0
            self._mapped_calls = 0
            self._declines.clear()
            self._group_counts.clear()
            self._support_counts.clear()
        if partitioned_calls or mapped_calls or declines:
            LOGGER.info(
                "Regional Linear route aggregate: mapped_calls=%d "
                "partitioned_calls=%d "
                "declines=%s group_counts=%s common_support_counts=%s",
                mapped_calls,
                partitioned_calls,
                declines,
                group_counts,
                most_common_support,
            )


REGIONAL_LINEAR_PERFORMANCE_DIAGNOSTICS = RegionalLinearPerformanceDiagnostics()
