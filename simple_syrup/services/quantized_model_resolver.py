# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve original checkpoints or globally cached quantized derivatives."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

from ..domain.model_quantization import ModelQuantizationRecipe, QuantizationProfile
from ..domain.quant_cache import QuantCacheIdentity, SourceCheckpointIdentity
from ..runtime.checkpoint_quantizer import SafetensorsCheckpointQuantizer
from ..runtime.quant_cache_leases import (
    GLOBAL_QUANT_CACHE_LEASES,
    QuantCacheLeaseRegistry,
)
from ..runtime.quant_cache_lock import QuantCacheBuildCoordinator
from ..runtime.quant_cache_repository import QuantCacheArtifact, QuantCacheRepository
from ..runtime.quant_cache_settings import SettingsQuantCacheLimitProvider
from ..runtime.quantization_capabilities import QuantizationCapabilityCatalog
from ..runtime.quantization_progress import (
    NullQuantizationProgressReporter,
    QuantizationProgressReporter,
)
from ..shared.logging import get_logger
from .quant_cache_service import QuantCacheService
from .quantized_model_boundaries import (
    CheckpointQuantizerBoundary,
    QuantCacheLimitProvider,
    QuantizationCapabilityBoundary,
    ResolvedModelCheckpoint,
)

LOGGER = get_logger(__name__)
HASH_CHUNK_SIZE = 8 * 1024 * 1024
MINIMUM_DISK_HEADROOM = 512 * 1024 * 1024


class QuantizedModelResolver:
    """Coordinate capability checks, generation, publication, and global LRU."""

    def __init__(
        self,
        repository: QuantCacheRepository | None = None,
        quantizer: CheckpointQuantizerBoundary | None = None,
        capabilities: QuantizationCapabilityBoundary | None = None,
        cache_service: QuantCacheService | None = None,
        limit_provider: QuantCacheLimitProvider | None = None,
        coordinator: QuantCacheBuildCoordinator | None = None,
        leases: QuantCacheLeaseRegistry | None = None,
    ) -> None:
        """Create a resolver with injectable architecture boundaries."""

        self._repository = repository or QuantCacheRepository()
        self._leases = leases or GLOBAL_QUANT_CACHE_LEASES
        self._quantizer = quantizer or SafetensorsCheckpointQuantizer()
        self._capabilities = capabilities or QuantizationCapabilityCatalog()
        self._cache_service = cache_service or QuantCacheService(
            self._repository, self._leases
        )
        self._limit_provider = limit_provider or SettingsQuantCacheLimitProvider()
        self._coordinator = coordinator or QuantCacheBuildCoordinator(
            self._repository.lock_directory
        )

    def resolve(
        self,
        *,
        source_model: str,
        source_path: Path,
        quantization: str,
        recipe: ModelQuantizationRecipe,
        progress: QuantizationProgressReporter | None = None,
    ) -> ResolvedModelCheckpoint:
        """Return the original path or create and return its cached derivative."""

        profile = recipe.profile_from_selection(quantization)
        if profile.is_original:
            return ResolvedModelCheckpoint(source_path, profile)
        self._capabilities.require_available(profile)

        resolved_source = source_path.resolve()
        source_stat = resolved_source.stat()
        cached = self._repository.find_current(
            source_model=source_model,
            source_path=resolved_source,
            source_size_bytes=source_stat.st_size,
            source_modified_ns=source_stat.st_mtime_ns,
            profile=profile,
            recipe=recipe,
        )
        if cached is not None:
            LOGGER.debug(
                "using cached quantized checkpoint",
                extra={
                    "source_model": source_model,
                    "quantization_profile": profile.profile_id,
                },
            )
            return self._cached_resolution(cached, profile)

        reporter = progress or NullQuantizationProgressReporter()
        total_work = max(source_stat.st_size * 2, 1)
        reporter.start(
            f"Creating {profile.label} cache for {source_model}",
            total_work,
        )
        source_sha256 = _sha256_with_progress(
            resolved_source,
            reporter,
            total_work,
        )
        identity = QuantCacheIdentity(
            source=SourceCheckpointIdentity(
                display_name=source_model,
                path=resolved_source,
                size_bytes=source_stat.st_size,
                modified_ns=source_stat.st_mtime_ns,
                sha256=source_sha256,
            ),
            profile=profile,
            model_family=recipe.model_family,
            recipe_version=recipe.version,
        )

        with self._coordinator.acquire(identity.stable_key):
            exact_cached = self._repository.find_identity(identity)
            if exact_cached is not None:
                reporter.finish()
                return self._cached_resolution(exact_cached, profile)
            self._require_disk_space(identity)
            started = time.monotonic()
            LOGGER.info(
                "creating %s cache for %s",
                profile.label,
                source_model,
                extra={
                    "source_model": source_model,
                    "quantization_profile": profile.profile_id,
                    "model_family": recipe.model_family,
                    "recipe_version": recipe.version,
                },
            )
            build_directory = self._repository.create_build_directory(identity)
            try:
                destination = build_directory / self._repository.artifact_filename(
                    identity
                )
                result = self._quantizer.quantize(
                    source=identity.source,
                    destination_path=destination,
                    profile=profile,
                    recipe=recipe,
                    progress=reporter,
                    progress_base=source_stat.st_size,
                    progress_total=total_work,
                )
                artifact = self._repository.commit(identity, build_directory)
            except Exception:
                self._repository.discard_build(build_directory)
                LOGGER.exception(
                    "quantized checkpoint generation failed",
                    extra={
                        "source_model": source_model,
                        "quantization_profile": profile.profile_id,
                    },
                )
                raise

        reporter.finish()
        limit_bytes = self._limit_provider.limit_bytes()
        eviction = self._cache_service.enforce_limit(
            limit_bytes,
            protected_paths=frozenset({artifact.path}),
        )
        if eviction.remaining_bytes > limit_bytes:
            LOGGER.warning(
                "quant cache remains above configured limit because the current "
                "or loaded artifacts are protected",
                extra={
                    "remaining_bytes": eviction.remaining_bytes,
                    "limit_bytes": limit_bytes,
                },
            )
        output_size = getattr(result, "output_size_bytes", artifact.path.stat().st_size)
        LOGGER.info(
            "created %s cache for %s in %.2f seconds",
            profile.label,
            source_model,
            time.monotonic() - started,
            extra={
                "source_model": source_model,
                "quantization_profile": profile.profile_id,
                "duration_seconds": round(time.monotonic() - started, 2),
                "artifact_size_bytes": output_size,
            },
        )
        return self._cached_resolution(artifact, profile)

    def _cached_resolution(
        self,
        artifact: QuantCacheArtifact,
        profile: QuantizationProfile,
    ) -> ResolvedModelCheckpoint:
        """Reserve a cache hit until its consumer establishes a model lease."""

        return ResolvedModelCheckpoint(
            artifact.path,
            profile,
            artifact,
            self._leases.reserve(artifact.path),
        )

    def _require_disk_space(self, identity: QuantCacheIdentity) -> None:
        """Fail before conversion when the cache drive lacks safe working space."""

        ratio = (
            0.55
            if any(item.value == "nvfp4" for item in identity.profile.required_formats)
            else 0.75
        )
        required = int(identity.source.size_bytes * ratio) + MINIMUM_DISK_HEADROOM
        self._repository.ensure_root()
        available = shutil.disk_usage(self._repository.root).free
        if available < required:
            raise OSError(
                f"Not enough free disk space to create the "
                f"{identity.profile.label} cache. Approximately "
                f"{required / 1024**3:.1f} GiB is required, but only "
                f"{available / 1024**3:.1f} GiB is available on the cache drive."
            )


def _sha256_with_progress(
    path: Path,
    progress: QuantizationProgressReporter,
    total_work: int,
) -> str:
    """Hash a source file incrementally while updating node progress."""

    digest = hashlib.sha256()
    processed = 0
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
            processed += len(chunk)
            progress.advance(processed, total_work)
    return digest.hexdigest()
