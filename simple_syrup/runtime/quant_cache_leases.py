# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Protect cache artifacts while loaded ComfyUI model objects reference them."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QuantCacheReservation:
    """Hold a temporary artifact lease across resolution and model loading."""

    artifact_path: Path
    _release_callback: Callable[[Path], None]
    _released: bool = field(default=False, init=False)

    def release(self) -> None:
        """Release the reservation exactly once."""

        if self._released:
            return
        self._released = True
        self._release_callback(self.artifact_path)


class QuantCacheLeaseRegistry:
    """Track artifact leases against the lifetime of underlying model objects."""

    def __init__(self) -> None:
        """Create an empty thread-safe lease registry."""

        self._counts: dict[Path, int] = {}
        self._owners: weakref.WeakKeyDictionary[object, set[Path]] = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.RLock()

    def lease(self, artifact_path: Path, loaded_model: object) -> None:
        """Protect an artifact until the underlying model is garbage collected."""

        owner = getattr(loaded_model, "model", loaded_model)
        normalized_path = artifact_path.resolve()
        with self._lock:
            existing = self._owners.setdefault(owner, set())
            if normalized_path in existing:
                return
            existing.add(normalized_path)
            self._acquire(normalized_path)
            weakref.finalize(owner, self._release, normalized_path)

    def reserve(self, artifact_path: Path) -> QuantCacheReservation:
        """Protect an artifact during work that precedes a model-owned lease."""

        normalized_path = artifact_path.resolve()
        with self._lock:
            self._acquire(normalized_path)
        return QuantCacheReservation(normalized_path, self._release)

    def is_leased(self, artifact_path: Path) -> bool:
        """Return whether a live model currently references the artifact."""

        with self._lock:
            return self._counts.get(artifact_path.resolve(), 0) > 0

    def active_count(self) -> int:
        """Return the number of distinct protected artifact paths."""

        with self._lock:
            return sum(count > 0 for count in self._counts.values())

    def _release(self, artifact_path: Path) -> None:
        """Release one model-owned artifact lease."""

        with self._lock:
            remaining = self._counts.get(artifact_path, 0) - 1
            if remaining > 0:
                self._counts[artifact_path] = remaining
            else:
                self._counts.pop(artifact_path, None)

    def _acquire(self, artifact_path: Path) -> None:
        """Increment one normalized path while the registry lock is held."""

        self._counts[artifact_path] = self._counts.get(artifact_path, 0) + 1


GLOBAL_QUANT_CACHE_LEASES = QuantCacheLeaseRegistry()
