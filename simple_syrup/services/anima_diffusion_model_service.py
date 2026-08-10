# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve and load Anima's original or recipe-quantized diffusion model."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from ..domain.anima_quantization import AnimaQuantizationRecipe
from ..runtime.diffusion_model_loader import DiffusionModelLoader
from ..runtime.quant_cache_leases import (
    GLOBAL_QUANT_CACHE_LEASES,
    QuantCacheLeaseRegistry,
)
from ..runtime.quant_cache_repository import QuantCacheRepository
from ..runtime.quantization_progress import QuantizationProgressReporter
from .quantized_model_boundaries import (
    DiffusionModelPathLoaderBoundary,
    QuantizedModelResolverBoundary,
)
from .quantized_model_resolver import QuantizedModelResolver


class AnimaDiffusionModelService:
    """Apply Anima policy while reusing global quantization infrastructure."""

    def __init__(
        self,
        diffusion_loader: DiffusionModelPathLoaderBoundary | None = None,
        resolver: QuantizedModelResolverBoundary | None = None,
        leases: QuantCacheLeaseRegistry | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a service with injectable loader, resolver, and lease owners."""

        self._diffusion_loader = diffusion_loader or DiffusionModelLoader(
            folder_paths_module
        )
        self._resolver = resolver or QuantizedModelResolver(
            repository=QuantCacheRepository(folder_paths_module=folder_paths_module)
        )
        self._leases = leases or GLOBAL_QUANT_CACHE_LEASES
        self._recipe = AnimaQuantizationRecipe()

    def load(
        self,
        *,
        diffusion_model: str,
        diffusion_weight_dtype: str,
        quantization: str,
        progress: QuantizationProgressReporter | None = None,
    ) -> object:
        """Load the selected source or a globally cached Anima derivative."""

        source_path = self._diffusion_loader.resolve_path(diffusion_model)
        resolved = self._resolver.resolve(
            source_model=diffusion_model,
            source_path=source_path,
            quantization=quantization,
            recipe=self._recipe,
            progress=progress,
        )
        effective_weight_dtype = (
            diffusion_weight_dtype if resolved.profile.is_original else "default"
        )
        try:
            model = self._diffusion_loader.load_path(
                resolved.path,
                effective_weight_dtype,
            )
            if resolved.cache_artifact is not None:
                self._leases.lease(resolved.cache_artifact.path, model)
        finally:
            if resolved.reservation is not None:
                resolved.reservation.release()
        model_boundary: Any = model
        model_boundary.simple_syrup_model_provenance = {
            "source_model": diffusion_model,
            "quantization_profile": resolved.profile.profile_id,
            "quantization_profile_label": resolved.profile.label,
            "quantization_profile_version": resolved.profile.version,
            "derived_path": str(resolved.path)
            if resolved.cache_artifact is not None
            else None,
        }
        return model
