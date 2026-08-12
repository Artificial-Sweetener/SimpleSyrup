# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for readable quant cache persistence, leases, and global LRU policy."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from simple_syrup.domain.anima_quantization import (
    MXFP8_PROFILE,
    NVFP4_MIXED_PROFILE,
    AnimaQuantizationRecipe,
)
from simple_syrup.domain.quant_cache import (
    QuantCacheIdentity,
    SourceCheckpointIdentity,
)
from simple_syrup.runtime import quant_cache_repository as repository_module
from simple_syrup.runtime.quant_cache_leases import QuantCacheLeaseRegistry
from simple_syrup.runtime.quant_cache_repository import (
    MANIFEST_FILENAME,
    README_FILENAME,
    QuantCacheArtifact,
    QuantCacheRepository,
)
from simple_syrup.services.quant_cache_service import QuantCacheService


def test_repository_creates_a_readable_global_cache_layout(tmp_path: Path) -> None:
    """Users browsing models/SyrupQuants can identify ownership and artifacts."""

    repository = QuantCacheRepository(tmp_path / "models" / "SyrupQuants")
    identity = _identity(tmp_path, "Anima/base model.safetensors", b"source-a")

    artifact = _commit(repository, identity, b"quantized")

    assert (
        (repository.root / README_FILENAME)
        .read_text(encoding="utf-8")
        .startswith("SimpleSyrup Quantized Model Cache")
    )
    assert artifact.path.relative_to(repository.root).as_posix() == (
        "Anima/base_model/nvfp4-mixed/"
        f"{identity.source.sha256[:12]}-profile-3-recipe-2/"
        "base_model--nvfp4-mixed.safetensors"
    )
    payload = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert payload["source_model"] == "Anima/base model.safetensors"
    assert payload["profile_id"] == "nvfp4-mixed"
    assert payload["profile_version"] == 3
    assert payload["quantization_formats"] == ["float8_e4m3fn", "nvfp4"]
    assert repository.relative_display_path() == "models/SyrupQuants"


def test_repository_finds_only_an_unchanged_source_signature(tmp_path: Path) -> None:
    """Fast cache hits invalidate when the authoritative source file changes."""

    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    identity = _identity(tmp_path, "anima.safetensors", b"source")
    artifact = _commit(repository, identity, b"quant")

    found = repository.find_current(
        source_model=identity.source.display_name,
        source_path=identity.source.path,
        source_size_bytes=identity.source.size_bytes,
        source_modified_ns=identity.source.modified_ns,
        profile=identity.profile,
        recipe=AnimaQuantizationRecipe(),
    )
    changed = repository.find_current(
        source_model=identity.source.display_name,
        source_path=identity.source.path,
        source_size_bytes=identity.source.size_bytes + 1,
        source_modified_ns=identity.source.modified_ns,
        profile=identity.profile,
        recipe=AnimaQuantizationRecipe(),
    )

    assert found is not None
    assert found.path == artifact.path
    assert changed is None


def test_profiles_have_isolated_readable_cache_identities(tmp_path: Path) -> None:
    """Mixed and MXFP8 recipes never share a path or cache artifact."""

    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    mixed = _identity(tmp_path, "anima.safetensors", b"source")
    mxfp8 = replace(mixed, profile=MXFP8_PROFILE)

    assert mixed.stable_key != mxfp8.stable_key
    assert repository.artifact_directory(mixed) != repository.artifact_directory(mxfp8)


def test_global_lru_preserves_reserved_artifacts_and_clears_inactive(
    tmp_path: Path,
) -> None:
    """Cache clearing never deletes an artifact reserved for active model work."""

    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    old = _commit(
        repository,
        _identity(tmp_path, "old.safetensors", b"old"),
        b"old-quant",
    )
    new = _commit(
        repository,
        _identity(tmp_path, "new.safetensors", b"new"),
        b"new-quantized",
    )
    _set_last_used(old, "2026-01-01T00:00:00+00:00")
    _set_last_used(new, "2026-02-01T00:00:00+00:00")
    leases = QuantCacheLeaseRegistry()
    reservation = leases.reserve(old.path)
    service = QuantCacheService(repository, leases)

    result = service.clear_inactive()

    assert result.removed_artifacts == 1
    assert old.path.is_file()
    assert not new.path.exists()
    assert service.status().active_artifact_count == 1
    reservation.release()
    second = service.clear_inactive()
    assert second.removed_artifacts == 1
    assert not old.path.exists()


def test_lru_evicts_oldest_artifact_until_under_budget(tmp_path: Path) -> None:
    """The global byte budget removes least-recently-used artifacts first."""

    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    old = _commit(
        repository,
        _identity(tmp_path, "old.safetensors", b"old-source"),
        b"12345",
    )
    new = _commit(
        repository,
        _identity(tmp_path, "new.safetensors", b"new-source"),
        b"1234567",
    )
    _set_last_used(old, "2026-01-01T00:00:00+00:00")
    _set_last_used(new, "2026-02-01T00:00:00+00:00")

    result = QuantCacheService(repository).enforce_limit(7)

    assert result.removed_artifacts == 1
    assert not old.path.exists()
    assert new.path.is_file()
    assert result.remaining_bytes == 7


def test_v1_artifact_is_never_reused_but_remains_clearable(tmp_path: Path) -> None:
    """Legacy soft quants participate in cleanup without matching v2 profiles."""

    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    identity = _identity(tmp_path, "anima.safetensors", b"source")
    legacy_directory = repository.root / "Anima" / "anima" / "nvfp4" / "legacy"
    legacy_directory.mkdir(parents=True)
    artifact_path = legacy_directory / "anima--nvfp4.safetensors"
    artifact_path.write_bytes(b"legacy")
    legacy_payload = {
        "schema_version": 1,
        "managed_by": "SimpleSyrup",
        "source_model": identity.source.display_name,
        "source_path": str(identity.source.path),
        "source_sha256": identity.source.sha256,
        "source_size_bytes": identity.source.size_bytes,
        "source_modified_ns": identity.source.modified_ns,
        "quantization_format": "nvfp4",
        "model_family": "Anima",
        "recipe_version": 1,
        "artifact_file": artifact_path.name,
        "artifact_size_bytes": artifact_path.stat().st_size,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_used_at": "2026-01-01T00:00:00+00:00",
    }
    (legacy_directory / MANIFEST_FILENAME).write_text(
        json.dumps(legacy_payload), encoding="utf-8"
    )

    assert (
        repository.find_current(
            source_model=identity.source.display_name,
            source_path=identity.source.path,
            source_size_bytes=identity.source.size_bytes,
            source_modified_ns=identity.source.modified_ns,
            profile=identity.profile,
            recipe=AnimaQuantizationRecipe(),
        )
        is None
    )
    assert len(repository.list_artifacts()) == 1
    assert QuantCacheService(repository).clear_inactive().removed_artifacts == 1
    assert not artifact_path.exists()


def test_manifest_replace_retries_one_transient_windows_sharing_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve atomic publication when a reader briefly holds the destination."""

    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    artifact = _commit(
        repository,
        _identity(tmp_path, "anima.safetensors", b"source"),
        b"quantized",
    )
    original_replace = Path.replace
    attempts = 0
    sleeps: list[float] = []

    def replace(source: Path, target: Path) -> Path:
        """Fail once like Windows, then execute the real atomic replacement."""

        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(13, "Access is denied", str(target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    touched = repository.touch(artifact)

    assert attempts == 2
    assert sleeps == [repository_module._MANIFEST_REPLACE_RETRY_SECONDS]
    assert touched.manifest_path.is_file()


def test_manifest_replace_propagates_exhausted_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when the destination never becomes atomically replaceable."""

    repository = QuantCacheRepository(tmp_path / "SyrupQuants")
    artifact = _commit(
        repository,
        _identity(tmp_path, "anima.safetensors", b"source"),
        b"quantized",
    )
    attempts = 0

    def deny(source: Path, target: Path) -> Path:
        """Keep the destination unavailable through every bounded attempt."""

        del source
        nonlocal attempts
        attempts += 1
        raise PermissionError(13, "Access is denied", str(target))

    monkeypatch.setattr(Path, "replace", deny)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    with pytest.raises(PermissionError, match="Access is denied"):
        repository.touch(artifact)

    assert attempts == repository_module._MANIFEST_REPLACE_ATTEMPTS
    assert list(artifact.manifest_path.parent.glob("*.tmp")) == []


def _identity(
    tmp_path: Path,
    display_name: str,
    content: bytes,
) -> QuantCacheIdentity:
    """Create an identity backed by an authoritative test source file."""

    source_path = tmp_path / f"source-{hashlib.sha256(content).hexdigest()[:8]}.bin"
    source_path.write_bytes(content)
    stat = source_path.stat()
    return QuantCacheIdentity(
        source=SourceCheckpointIdentity(
            display_name=display_name,
            path=source_path.resolve(),
            size_bytes=stat.st_size,
            modified_ns=stat.st_mtime_ns,
            sha256=hashlib.sha256(content).hexdigest(),
        ),
        profile=NVFP4_MIXED_PROFILE,
        model_family="Anima",
        recipe_version=2,
    )


def _commit(
    repository: QuantCacheRepository,
    identity: QuantCacheIdentity,
    content: bytes,
) -> QuantCacheArtifact:
    """Publish one tiny managed artifact through the production repository."""

    build = repository.create_build_directory(identity)
    (build / repository.artifact_filename(identity)).write_bytes(content)
    return repository.commit(identity, build)


def _set_last_used(artifact: QuantCacheArtifact, timestamp: str) -> None:
    """Set deterministic LRU order in one human-readable manifest."""

    manifest = replace(artifact.manifest, last_used_at=timestamp)
    artifact.manifest_path.write_text(
        json.dumps(manifest.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert artifact.manifest_path.name == MANIFEST_FILENAME
