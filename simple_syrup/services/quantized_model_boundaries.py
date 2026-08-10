# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define collaboration boundaries for reusable quantized model resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..domain.model_quantization import ModelQuantizationRecipe, QuantizationProfile
from ..domain.quant_cache import SourceCheckpointIdentity
from ..runtime.quant_cache_leases import QuantCacheReservation
from ..runtime.quant_cache_repository import QuantCacheArtifact
from ..runtime.quantization_progress import QuantizationProgressReporter


@dataclass(frozen=True)
class ResolvedModelCheckpoint:
    """Describe the actual checkpoint selected for one loader execution."""

    path: Path
    profile: QuantizationProfile
    cache_artifact: QuantCacheArtifact | None = None
    reservation: QuantCacheReservation | None = None


class CheckpointQuantizerBoundary(Protocol):
    """Convert one source checkpoint into a ComfyUI quantized checkpoint."""

    def quantize(
        self,
        *,
        source: SourceCheckpointIdentity,
        destination_path: Path,
        profile: QuantizationProfile,
        recipe: ModelQuantizationRecipe,
        progress: QuantizationProgressReporter,
        progress_base: int,
        progress_total: int,
    ) -> object:
        """Write and validate one derived checkpoint."""


class QuantCacheLimitProvider(Protocol):
    """Provide the current global quant cache byte limit."""

    def limit_bytes(self) -> int:
        """Return the validated global byte budget."""


class QuantizationCapabilityBoundary(Protocol):
    """Validate requested formats against ComfyUI and the active GPU."""

    def require_available(self, profile: QuantizationProfile) -> None:
        """Reject an unavailable profile."""


class DiffusionModelPathLoaderBoundary(Protocol):
    """Resolve and load standalone diffusion checkpoints by path."""

    def resolve_path(self, diffusion_model: str) -> Path:
        """Resolve one workflow model name to its source path."""

    def load_path(self, model_path: Path, weight_dtype: str) -> object:
        """Load one resolved diffusion checkpoint."""


class QuantizedModelResolverBoundary(Protocol):
    """Resolve source checkpoints to original or cached derivative paths."""

    def resolve(
        self,
        *,
        source_model: str,
        source_path: Path,
        quantization: str,
        recipe: ModelQuantizationRecipe,
        progress: QuantizationProgressReporter | None = None,
    ) -> ResolvedModelCheckpoint:
        """Return one protected model checkpoint resolution."""
