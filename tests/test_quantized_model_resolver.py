# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for reusable quantized checkpoint resolution and generation."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from simple_syrup.domain.anima_quantization import (
    NVFP4_MIXED_PROFILE,
    AnimaQuantizationRecipe,
)
from simple_syrup.domain.model_quantization import (
    ModelQuantizationRecipe,
    QuantizationFormat,
    QuantizationProfile,
    TensorDescriptor,
)
from simple_syrup.domain.quant_cache import SourceCheckpointIdentity
from simple_syrup.runtime.quant_cache_leases import QuantCacheLeaseRegistry
from simple_syrup.runtime.quant_cache_repository import QuantCacheRepository
from simple_syrup.services.quantized_model_resolver import QuantizedModelResolver


class RecordingCapabilities:
    """Accept and record requested test formats."""

    def __init__(self) -> None:
        """Create an empty request list."""

        self.requests: list[QuantizationProfile] = []

    def require_available(self, profile: QuantizationProfile) -> None:
        """Record an accepted profile."""

        self.requests.append(profile)


class FixedLimitProvider:
    """Return a deterministic generous cache budget."""

    def limit_bytes(self) -> int:
        """Return one GiB for tiny test artifacts."""

        return 1024**3


@dataclass(frozen=True)
class FakeQuantizationResult:
    """Expose output size through the quantizer result boundary."""

    output_size_bytes: int


class RecordingQuantizer:
    """Write a tiny artifact and record conversion calls."""

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        """Create a fake with optional concurrency delay or failure."""

        self.delay = delay
        self.fail = fail
        self.calls: list[SourceCheckpointIdentity] = []
        self._lock = threading.Lock()

    def quantize(
        self,
        *,
        source: SourceCheckpointIdentity,
        destination_path: Path,
        profile: QuantizationProfile,
        recipe: ModelQuantizationRecipe,
        progress: object,
        progress_base: int,
        progress_total: int,
    ) -> FakeQuantizationResult:
        """Write deterministic bytes through the production build directory."""

        del profile, recipe, progress, progress_base, progress_total
        with self._lock:
            self.calls.append(source)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("conversion failed")
        destination_path.write_bytes(b"quantized")
        return FakeQuantizationResult(destination_path.stat().st_size)


@dataclass
class RecordingProgress:
    """Record resolver-owned progress lifecycle and absolute updates."""

    starts: list[tuple[str, int]] = field(default_factory=list)
    updates: list[tuple[int, int]] = field(default_factory=list)
    finishes: int = 0

    def start(self, label: str, total: int) -> None:
        """Record operation start."""

        self.starts.append((label, total))

    def advance(self, current: int, total: int) -> None:
        """Record one absolute update."""

        self.updates.append((current, total))

    def finish(self) -> None:
        """Record operation completion."""

        self.finishes += 1


@dataclass(frozen=True)
class AlternateLoaderRecipe:
    """Represent a future loader's independently versioned quantization policy."""

    model_family: str = "FutureLoader"
    version: int = 3

    @property
    def profiles(self) -> tuple[QuantizationProfile, ...]:
        """Return the illustrative loader's supported profiles."""

        return (NVFP4_MIXED_PROFILE,)

    def profile_from_selection(self, selection: str) -> QuantizationProfile:
        """Parse the illustrative loader's sole profile."""

        if selection in (NVFP4_MIXED_PROFILE.label, NVFP4_MIXED_PROFILE.profile_id):
            return NVFP4_MIXED_PROFILE
        raise ValueError("unsupported profile")

    def policy_for(
        self,
        tensor: TensorDescriptor,
        profile: QuantizationProfile,
    ) -> QuantizationFormat | None:
        """Select matrices for the illustrative future loader."""

        del profile
        return QuantizationFormat.NVFP4 if len(tensor.shape) == 2 else None


def test_resolver_generates_once_then_uses_the_global_cache(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cache miss converts once and later unchanged requests are cache hits."""

    source = tmp_path / "anima.safetensors"
    source.write_bytes(b"authoritative-source")
    repository = QuantCacheRepository(tmp_path / "models" / "SyrupQuants")
    quantizer = RecordingQuantizer()
    capabilities = RecordingCapabilities()
    leases = QuantCacheLeaseRegistry()
    resolver = QuantizedModelResolver(
        repository=repository,
        quantizer=quantizer,
        capabilities=capabilities,
        limit_provider=FixedLimitProvider(),
        leases=leases,
    )
    first_progress = RecordingProgress()

    with caplog.at_level(logging.INFO):
        first = resolver.resolve(
            source_model="Anima/anima.safetensors",
            source_path=source,
            quantization="nvfp4-mixed",
            recipe=AnimaQuantizationRecipe(),
            progress=first_progress,
        )
    assert first.reservation is not None
    first.reservation.release()
    second_progress = RecordingProgress()
    with caplog.at_level(logging.INFO):
        second = resolver.resolve(
            source_model="Anima/anima.safetensors",
            source_path=source,
            quantization="nvfp4-mixed",
            recipe=AnimaQuantizationRecipe(),
            progress=second_progress,
        )
    assert second.reservation is not None
    second.reservation.release()

    assert first.path == second.path
    assert len(quantizer.calls) == 1
    assert capabilities.requests == [
        NVFP4_MIXED_PROFILE,
        NVFP4_MIXED_PROFILE,
    ]
    assert len(first_progress.starts) == 1
    assert first_progress.finishes == 1
    assert second_progress.starts == []
    assert (repository.root / "README.txt").is_file()
    resolver_info = [
        record.message
        for record in caplog.records
        if record.name.endswith("quantized_model_resolver")
        and record.levelno == logging.INFO
    ]
    assert resolver_info[0] == (
        "creating NVFP4 (Mixed) cache for Anima/anima.safetensors"
    )
    assert resolver_info[1].startswith(
        "created NVFP4 (Mixed) cache for Anima/anima.safetensors in "
    )
    assert len(resolver_info) == 2


def test_original_resolution_does_not_read_or_create_the_cache(tmp_path: Path) -> None:
    """Original selection returns its path without touching a missing source file."""

    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    resolver = QuantizedModelResolver(
        repository=repository,
        quantizer=RecordingQuantizer(),
        capabilities=RecordingCapabilities(),
        limit_provider=FixedLimitProvider(),
    )
    missing_source = tmp_path / "not-read.safetensors"

    resolved = resolver.resolve(
        source_model="not-read.safetensors",
        source_path=missing_source,
        quantization="Original",
        recipe=AnimaQuantizationRecipe(),
    )

    assert resolved.path == missing_source
    assert resolved.cache_artifact is None
    assert not repository.root.exists()


def test_model_family_and_recipe_version_are_part_of_cache_identity(
    tmp_path: Path,
) -> None:
    """Future loaders can reuse infrastructure without sharing incompatible quants."""

    source = tmp_path / "shared.safetensors"
    source.write_bytes(b"shared-authoritative-source")
    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    quantizer = RecordingQuantizer()
    resolver = QuantizedModelResolver(
        repository=repository,
        quantizer=quantizer,
        capabilities=RecordingCapabilities(),
        limit_provider=FixedLimitProvider(),
    )

    anima = resolver.resolve(
        source_model="shared.safetensors",
        source_path=source,
        quantization="nvfp4-mixed",
        recipe=AnimaQuantizationRecipe(),
        progress=RecordingProgress(),
    )
    future = resolver.resolve(
        source_model="shared.safetensors",
        source_path=source,
        quantization="nvfp4-mixed",
        recipe=AlternateLoaderRecipe(),
        progress=RecordingProgress(),
    )
    assert anima.reservation is not None
    assert future.reservation is not None
    anima.reservation.release()
    future.reservation.release()

    assert anima.path != future.path
    assert "Anima" in anima.path.parts
    assert "FutureLoader" in future.path.parts
    assert len(quantizer.calls) == 2


def test_failed_generation_removes_partial_build_directory(tmp_path: Path) -> None:
    """Conversion failures leave no valid-looking partial cache artifacts."""

    source = tmp_path / "anima.safetensors"
    source.write_bytes(b"source")
    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    resolver = QuantizedModelResolver(
        repository=repository,
        quantizer=RecordingQuantizer(fail=True),
        capabilities=RecordingCapabilities(),
        limit_provider=FixedLimitProvider(),
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        resolver.resolve(
            source_model="anima.safetensors",
            source_path=source,
            quantization="nvfp4-mixed",
            recipe=AnimaQuantizationRecipe(),
            progress=RecordingProgress(),
        )

    building_root = repository.root / ".building"
    assert not building_root.exists() or list(building_root.iterdir()) == []
    assert repository.list_artifacts() == ()


def test_concurrent_requests_share_one_generated_artifact(tmp_path: Path) -> None:
    """Thread and file locks prevent duplicate work for the same cache identity."""

    source = tmp_path / "anima.safetensors"
    source.write_bytes(b"same-source")
    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    quantizer = RecordingQuantizer(delay=0.2)
    leases = QuantCacheLeaseRegistry()
    resolver = QuantizedModelResolver(
        repository=repository,
        quantizer=quantizer,
        capabilities=RecordingCapabilities(),
        limit_provider=FixedLimitProvider(),
        leases=leases,
    )
    results: list[Path] = []
    failures: list[BaseException] = []
    results_lock = threading.Lock()

    def resolve_once() -> None:
        """Resolve and release one temporary cache reservation."""

        try:
            resolved = resolver.resolve(
                source_model="anima.safetensors",
                source_path=source,
                quantization="nvfp4-mixed",
                recipe=AnimaQuantizationRecipe(),
                progress=RecordingProgress(),
            )
            assert resolved.reservation is not None
            resolved.reservation.release()
            with results_lock:
                results.append(resolved.path)
        except BaseException as error:
            with results_lock:
                failures.append(error)

    threads = [threading.Thread(target=resolve_once) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert failures == []
    assert len(results) == 2
    assert results[0] == results[1]
    assert len(quantizer.calls) == 1
