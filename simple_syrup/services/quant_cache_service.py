# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply global LRU policy to inactive SimpleSyrup quant cache artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..runtime.quant_cache_leases import (
    GLOBAL_QUANT_CACHE_LEASES,
    QuantCacheLeaseRegistry,
)
from ..runtime.quant_cache_repository import QuantCacheArtifact, QuantCacheRepository
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class QuantCacheStatus:
    """Summarize the global cache for settings and diagnostics."""

    path: str
    usage_bytes: int
    artifact_count: int
    active_artifact_count: int

    def to_payload(self, limit_bytes: int) -> dict[str, object]:
        """Return the settings-route JSON representation."""

        return {
            "path": self.path,
            "usage_bytes": self.usage_bytes,
            "limit_bytes": limit_bytes,
            "artifact_count": self.artifact_count,
            "active_artifact_count": self.active_artifact_count,
        }


@dataclass(frozen=True)
class QuantCacheEvictionResult:
    """Summarize one bounded eviction attempt."""

    removed_artifacts: int
    removed_bytes: int
    remaining_bytes: int


class QuantCacheService:
    """Own global cache accounting, LRU eviction, and safe clearing."""

    def __init__(
        self,
        repository: QuantCacheRepository | None = None,
        leases: QuantCacheLeaseRegistry | None = None,
    ) -> None:
        """Create a cache service with injectable persistence and leases."""

        self._repository = repository or QuantCacheRepository()
        self._leases = leases or GLOBAL_QUANT_CACHE_LEASES

    def status(self) -> QuantCacheStatus:
        """Return current managed artifact usage."""

        artifacts = self._repository.list_artifacts()
        return QuantCacheStatus(
            path=self._repository.relative_display_path(),
            usage_bytes=sum(item.manifest.artifact_size_bytes for item in artifacts),
            artifact_count=len(artifacts),
            active_artifact_count=self._leases.active_count(),
        )

    def enforce_limit(
        self,
        limit_bytes: int,
        protected_paths: frozenset[Path] = frozenset(),
    ) -> QuantCacheEvictionResult:
        """Evict least-recently-used inactive artifacts until within the limit."""

        if limit_bytes < 0:
            raise ValueError("Quant cache limit must not be negative.")
        artifacts = list(self._repository.list_artifacts())
        remaining = sum(item.manifest.artifact_size_bytes for item in artifacts)
        removed_count = 0
        removed_bytes = 0
        protected = {path.resolve() for path in protected_paths}
        for artifact in sorted(artifacts, key=_last_used):
            if remaining <= limit_bytes:
                break
            if artifact.path.resolve() in protected or self._leases.is_leased(
                artifact.path
            ):
                continue
            if not self._repository.remove(artifact):
                continue
            removed_count += 1
            removed_bytes += artifact.manifest.artifact_size_bytes
            remaining -= artifact.manifest.artifact_size_bytes
        if removed_count:
            LOGGER.info(
                "quant cache eviction completed",
                extra={
                    "removed_artifacts": removed_count,
                    "removed_bytes": removed_bytes,
                    "remaining_bytes": remaining,
                    "limit_bytes": limit_bytes,
                },
            )
        return QuantCacheEvictionResult(removed_count, removed_bytes, remaining)

    def clear_inactive(self) -> QuantCacheEvictionResult:
        """Remove every inactive managed artifact and preserve loaded models."""

        return self.enforce_limit(0)


def _last_used(artifact: QuantCacheArtifact) -> float:
    """Return the manifest timestamp used for deterministic LRU ordering."""

    try:
        return datetime.fromisoformat(artifact.manifest.last_used_at).timestamp()
    except ValueError:
        return float("-inf")
