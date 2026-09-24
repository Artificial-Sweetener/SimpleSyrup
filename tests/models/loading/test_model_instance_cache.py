# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for process-level model instance cache mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from simple_syrup.runtime.model_instance_cache import ModelInstanceCache


@dataclass(frozen=True)
class FakeCacheKey:
    """Identify fake loaded objects in cache tests."""

    value: str


class LoadFailure(RuntimeError):
    """Distinct exception raised by fake loaders."""


def test_get_or_load_calls_loader_on_first_miss() -> None:
    """A cache miss stores and returns the loader result."""

    loaded = object()
    calls = 0

    def load() -> object:
        """Return a fixed fake loaded model."""

        nonlocal calls
        calls += 1
        return loaded

    cache: ModelInstanceCache[FakeCacheKey, object] = ModelInstanceCache()

    assert cache.get_or_load(FakeCacheKey("model"), load) is loaded
    assert calls == 1


def test_get_or_load_reuses_cached_instance() -> None:
    """A cache hit returns the original object without loading again."""

    first = object()
    calls = 0

    def load() -> object:
        """Return a new fake loaded model."""

        nonlocal calls
        calls += 1
        return first

    cache: ModelInstanceCache[FakeCacheKey, object] = ModelInstanceCache()

    assert cache.get_or_load(FakeCacheKey("model"), load) is first
    assert cache.get_or_load(FakeCacheKey("model"), load) is first
    assert calls == 1


def test_get_or_load_separates_different_keys() -> None:
    """Distinct keys keep distinct loaded instances."""

    calls = 0

    def load() -> object:
        """Return a unique fake loaded model."""

        nonlocal calls
        calls += 1
        return object()

    cache: ModelInstanceCache[FakeCacheKey, object] = ModelInstanceCache()

    first = cache.get_or_load(FakeCacheKey("first"), load)
    second = cache.get_or_load(FakeCacheKey("second"), load)

    assert second is not first
    assert calls == 2


def test_get_or_load_propagates_load_failures() -> None:
    """Loader exceptions surface to callers."""

    def load() -> object:
        """Raise a fake loading failure."""

        raise LoadFailure("failed")

    cache: ModelInstanceCache[FakeCacheKey, object] = ModelInstanceCache()

    with pytest.raises(LoadFailure, match="failed"):
        cache.get_or_load(FakeCacheKey("model"), load)


def test_get_or_load_does_not_cache_failed_loads() -> None:
    """A failed load leaves the key available for a later successful load."""

    calls = 0
    loaded = object()

    def load() -> object:
        """Fail once, then return a fake loaded model."""

        nonlocal calls
        calls += 1
        if calls == 1:
            raise LoadFailure("failed")
        return loaded

    cache: ModelInstanceCache[FakeCacheKey, object] = ModelInstanceCache()
    key = FakeCacheKey("model")

    with pytest.raises(LoadFailure, match="failed"):
        cache.get_or_load(key, load)

    assert cache.get_or_load(key, load) is loaded
    assert calls == 2
