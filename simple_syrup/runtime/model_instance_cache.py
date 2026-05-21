# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reuse loaded external model instances by resolved runtime identity."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT")
ModelT = TypeVar("ModelT")


@dataclass
class ModelInstanceCache(Generic[KeyT, ModelT]):
    """Coordinate process-level reuse for SimpleSyrup-owned model instances."""

    entries: MutableMapping[KeyT, ModelT] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def get_or_load(self, key: KeyT, load: Callable[[], ModelT]) -> ModelT:
        """Return a cached model instance or load and store it atomically."""

        with self._lock:
            cached = self.entries.get(key)
            if cached is not None:
                return cached
            loaded = load()
            self.entries[key] = loaded
            return loaded
