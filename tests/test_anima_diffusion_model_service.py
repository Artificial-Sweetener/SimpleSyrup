# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Anima's thin integration with reusable quantized model loading."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

from simple_syrup.domain.anima_quantization import NVFP4_MIXED_PROFILE
from simple_syrup.domain.model_quantization import (
    ModelQuantizationRecipe,
)
from simple_syrup.domain.quant_cache import QuantCacheManifest
from simple_syrup.runtime.quant_cache_leases import QuantCacheLeaseRegistry
from simple_syrup.runtime.quant_cache_repository import QuantCacheArtifact
from simple_syrup.runtime.quantization_progress import QuantizationProgressReporter
from simple_syrup.services.anima_diffusion_model_service import (
    AnimaDiffusionModelService,
)
from simple_syrup.services.quantized_model_boundaries import (
    ResolvedModelCheckpoint,
)


class FakeUnderlyingModel:
    """Weak-referenceable owner shared by a fake Comfy model patcher."""


class FakeModelPatcher:
    """Expose Comfy's underlying-model attribute used for cache leases."""

    def __init__(self) -> None:
        """Create one underlying model owner."""

        self.model = FakeUnderlyingModel()


class FakeDiffusionLoader:
    """Resolve the source and record actual checkpoint loading."""

    def __init__(self, source_path: Path) -> None:
        """Create a fake for one source path."""

        self.source_path = source_path
        self.load_calls: list[tuple[Path, str]] = []

    def resolve_path(self, diffusion_model: str) -> Path:
        """Return the configured authoritative source path."""

        del diffusion_model
        return self.source_path

    def load_path(self, model_path: Path, weight_dtype: str) -> object:
        """Record the selected runtime path and dtype."""

        self.load_calls.append((model_path, weight_dtype))
        return FakeModelPatcher()


@dataclass
class FakeResolver:
    """Return a predetermined protected checkpoint resolution."""

    resolved: ResolvedModelCheckpoint
    requested_source_model: str | None = None

    def resolve(
        self,
        *,
        source_model: str,
        source_path: Path,
        quantization: str,
        recipe: ModelQuantizationRecipe,
        progress: QuantizationProgressReporter | None = None,
    ) -> ResolvedModelCheckpoint:
        """Record source identity while returning the configured result."""

        del source_path, quantization, recipe, progress
        self.requested_source_model = source_model
        return self.resolved


def test_anima_loads_cached_derivative_with_source_provenance_and_lease(
    tmp_path: Path,
) -> None:
    """Runtime cache paths never replace the workflow's authoritative model name."""

    source_path = tmp_path / "models" / "diffusion_models" / "anima.safetensors"
    artifact_path = tmp_path / "models" / "SyrupQuants" / "Anima" / "cached.safetensors"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"quant")
    manifest_path = artifact_path.with_name("manifest.json")
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = QuantCacheManifest(
        source_model="Anima/anima.safetensors",
        source_path=str(source_path),
        source_sha256="a" * 64,
        source_size_bytes=10,
        source_modified_ns=1,
        profile_id="nvfp4-mixed",
        profile_label="NVFP4 (Mixed)",
        profile_version=3,
        quantization_formats=("float8_e4m3fn", "nvfp4"),
        model_family="Anima",
        recipe_version=2,
        artifact_file=artifact_path.name,
        artifact_size_bytes=5,
        created_at="2026-01-01T00:00:00+00:00",
        last_used_at="2026-01-01T00:00:00+00:00",
    )
    artifact = QuantCacheArtifact(artifact_path, manifest_path, manifest)
    leases = QuantCacheLeaseRegistry()
    resolver = FakeResolver(
        ResolvedModelCheckpoint(
            artifact_path,
            NVFP4_MIXED_PROFILE,
            artifact,
            leases.reserve(artifact_path),
        )
    )
    loader = FakeDiffusionLoader(source_path)
    service = AnimaDiffusionModelService(loader, resolver, leases)

    loaded = service.load(
        diffusion_model="Anima/anima.safetensors",
        diffusion_weight_dtype="fp8_e4m3fn_fast",
        quantization="nvfp4-mixed",
    )

    assert loader.load_calls == [(artifact_path, "default")]
    assert resolver.requested_source_model == "Anima/anima.safetensors"
    provenance = loaded.simple_syrup_model_provenance  # type: ignore[attr-defined]
    assert provenance["source_model"] == "Anima/anima.safetensors"
    assert provenance["quantization_profile"] == "nvfp4-mixed"
    assert provenance["quantization_profile_version"] == 3
    assert leases.is_leased(artifact_path)
    del loaded
    gc.collect()
    assert not leases.is_leased(artifact_path)
