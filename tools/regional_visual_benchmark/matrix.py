"""Define the immutable corrected P10.3 visual benchmark matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tools.attention_coupling_benchmark.manifest_types import BenchmarkManifest

SOURCE_WIDTH = 1024
SOURCE_HEIGHT = 1024
TARGET_WIDTH = 1536
TARGET_HEIGHT = 1536
REFINEMENT_DENOISE = 0.30
SPATIAL_SIZE = 96
SPATIAL_OVERLAP = 32
SPATIAL_BATCH_SIZE = 4
CONTEXTUAL_GLOBAL_STEPS = 3


class RegionalStrategy(StrEnum):
    """Name one explicit regional execution strategy."""

    REGIONAL_CONDITIONING = "regional_conditioning"
    ATTENTION_COUPLING = "attention_coupling"


class SpatialProfile(StrEnum):
    """Name one full or corrected refinement execution profile."""

    FULL = "full"
    TILED_MULTIDIFFUSION = "tiled_multidiffusion"
    TILED_MIXTURE_OF_DIFFUSERS = "tiled_mixture_of_diffusers"
    CONTEXTUAL_MULTIDIFFUSION = "contextual_multidiffusion"
    CONTEXTUAL_MIXTURE_OF_DIFFUSERS = "contextual_mixture_of_diffusers"

    @property
    def is_refinement(self) -> bool:
        """Return whether this profile consumes a shared 1024 source."""

        return self is not SpatialProfile.FULL

    @property
    def diffusion_mode(self) -> str | None:
        """Return the public tiled fusion value when applicable."""

        if self in {
            SpatialProfile.TILED_MULTIDIFFUSION,
            SpatialProfile.CONTEXTUAL_MULTIDIFFUSION,
        }:
            return "multidiffusion"
        if self in {
            SpatialProfile.TILED_MIXTURE_OF_DIFFUSERS,
            SpatialProfile.CONTEXTUAL_MIXTURE_OF_DIFFUSERS,
        }:
            return "mixture_of_diffusers"
        return None


@dataclass(frozen=True, slots=True)
class SourcePosition:
    """Identify one strategy-neutral source generation."""

    source_id: str
    case_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class VisualPosition:
    """Identify one mandatory strategy and spatial-profile output."""

    artifact_id: str
    case_id: str
    seed: int
    strategy: RegionalStrategy
    spatial_profile: SpatialProfile
    source_id: str | None

    @property
    def is_refinement(self) -> bool:
        """Return whether the position consumes a neutral source."""

        return self.source_id is not None

    @property
    def expected_size(self) -> tuple[int, int]:
        """Return the exact decoded output dimensions."""

        if self.is_refinement:
            return TARGET_WIDTH, TARGET_HEIGHT
        return SOURCE_WIDTH, SOURCE_HEIGHT


def source_positions(manifest: BenchmarkManifest) -> tuple[SourcePosition, ...]:
    """Expand one neutral source for every frozen case and seed."""

    return tuple(
        SourcePosition(
            source_id=f"source__{case.case_id}__seed-{seed}",
            case_id=case.case_id,
            seed=seed,
        )
        for case in manifest.cases
        for seed in manifest.sampling.seeds
    )


def visual_positions(manifest: BenchmarkManifest) -> tuple[VisualPosition, ...]:
    """Expand every case, seed, profile, and strategy exactly once."""

    profiles = tuple(SpatialProfile)
    strategies = tuple(RegionalStrategy)
    return tuple(
        _position(manifest.benchmark_id, case.case_id, seed, profile, strategy)
        for case in manifest.cases
        for seed in manifest.sampling.seeds
        for profile in profiles
        for strategy in strategies
    )


def _position(
    benchmark_id: str,
    case_id: str,
    seed: int,
    profile: SpatialProfile,
    strategy: RegionalStrategy,
) -> VisualPosition:
    """Build one stable matrix position without encoding display labels."""

    source_id = f"source__{case_id}__seed-{seed}" if profile.is_refinement else None
    artifact_id = (
        f"{benchmark_id}__{case_id}__{profile.value}__{strategy.value}__seed-{seed}"
    )
    return VisualPosition(
        artifact_id=artifact_id,
        case_id=case_id,
        seed=seed,
        strategy=strategy,
        spatial_profile=profile,
        source_id=source_id,
    )
