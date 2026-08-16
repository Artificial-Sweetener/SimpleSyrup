# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare the immutable P10.1 regional LoRA scaling matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .manifest import PerformanceArtifact


class PerformanceAttentionBackend(StrEnum):
    """Name each explicitly measured installed attention implementation."""

    PYTORCH = "pytorch"
    SAGE = "sage"


class PerformanceMaskLayout(StrEnum):
    """Name canonical mask geometry without owning projection behavior."""

    HALF_FIRST = "half_first"
    HARD_PARTITION = "hard_partition"
    FULL_FIRST = "full_first"
    REGION_ZERO_FULL = "region_zero_full"


class PerformanceAdapterLayout(StrEnum):
    """Name adapter identity, region, and schedule construction policy."""

    NONE = "none"
    DISTINCT_STACKED = "distinct_stacked"
    DISTINCT_PER_REGION = "distinct_per_region"
    REPEATED_PER_REGION = "repeated_per_region"
    INACTIVE_STACKED = "inactive_stacked"


class PerformanceEqualityMode(StrEnum):
    """Name absent, bit-exact, or bounded BF16 output comparison."""

    NONE = "none"
    EXACT = "exact"
    BF16 = "bf16"


@dataclass(frozen=True, slots=True)
class PerformanceWorkExpectation:
    """Declare exact production-owned cache and diagnostic cardinalities."""

    prepared_cache_entries: int
    active_adapter_uses: int
    active_target_count: int
    target_use_count: int
    deduplicated_target_group_count: int
    compatible_projection_batch_count: int
    deduplicated_target_uses: int


@dataclass(frozen=True, slots=True)
class ScalingPerformanceProfile:
    """Declare one deterministic region, adapter, backend, and acceptance point."""

    profile_id: str
    attention_backend: PerformanceAttentionBackend
    region_count: int
    mask_layout: PerformanceMaskLayout
    adapter_layout: PerformanceAdapterLayout
    adapter_count: int
    work: PerformanceWorkExpectation
    equality_profile_id: str | None
    equality_mode: PerformanceEqualityMode
    maximum_overhead_percent: float | None
    capture_operator_trace: bool

    def __post_init__(self) -> None:
        """Reject malformed profile cardinality before fixture construction."""

        if not self.profile_id:
            raise ValueError("Scaling profile id must be non-empty.")
        if self.region_count not in (1, 2, 4):
            raise ValueError("Scaling profile regions must be one, two, or four.")
        if self.adapter_count not in (0, 1, 2, 4):
            raise ValueError(
                "Scaling profile adapters must be zero, one, two, or four."
            )
        if (self.adapter_layout is PerformanceAdapterLayout.NONE) != (
            self.adapter_count == 0
        ):
            raise ValueError("Scaling adapter layout must agree with adapter count.")
        if self.maximum_overhead_percent is not None and (
            self.maximum_overhead_percent <= 0.0
        ):
            raise ValueError("Scaling overhead limits must be positive when present.")
        if (self.equality_profile_id is None) != (
            self.equality_mode is PerformanceEqualityMode.NONE
        ):
            raise ValueError("Scaling equality mode must agree with its reference.")


@dataclass(frozen=True, slots=True)
class ScalingPerformanceManifest:
    """Retain fixed geometry, trajectory, artifacts, and ordered positions."""

    benchmark_id: str
    width: int
    height: int
    latent_channels: int
    context_tokens: int
    context_features: int
    denoiser_calls: int
    warmup_calls: int
    repeats: int
    seed: int
    artifacts: tuple[PerformanceArtifact, ...]
    profiles: tuple[ScalingPerformanceProfile, ...]

    def __post_init__(self) -> None:
        """Require the complete P10.1 matrix and stable measurement controls."""

        if (self.width, self.height) != (1024, 1024):
            raise ValueError("P10.1 scaling geometry must be 1024x1024.")
        if self.denoiser_calls != 30:
            raise ValueError("P10.1 scaling requires 30 denoiser calls.")
        if self.warmup_calls != 2 or self.repeats < 3:
            raise ValueError("P10.1 scaling requires two warmups and three repeats.")
        profile_ids = tuple(profile.profile_id for profile in self.profiles)
        if profile_ids != _PROFILE_IDS:
            raise ValueError("P10.1 scaling profiles must retain the complete order.")
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("P10.1 scaling profile ids must be unique.")
        known_ids = set(profile_ids)
        if any(
            profile.equality_profile_id not in known_ids
            for profile in self.profiles
            if profile.equality_profile_id is not None
        ):
            raise ValueError("P10.1 equality controls must name declared profiles.")


_ZERO_WORK = PerformanceWorkExpectation(0, 0, 0, 0, 0, 0, 0)
_ONE_WORK = PerformanceWorkExpectation(448, 1, 448, 448, 448, 448, 0)
_TWO_WORK = PerformanceWorkExpectation(896, 2, 448, 896, 896, 448, 0)
_FOUR_WORK = PerformanceWorkExpectation(1792, 4, 448, 1792, 1792, 448, 0)
_REPEATED_WORK = PerformanceWorkExpectation(
    448,
    4,
    448,
    1792,
    448,
    448,
    1344,
)
_PROFILE_IDS = (
    "pytorch-r1-lora0",
    "pytorch-r1-lora1",
    "pytorch-r1-lora2-stacked",
    "pytorch-r1-lora4-stacked",
    "pytorch-r2-lora0",
    "pytorch-r4-lora0",
    "pytorch-r2-lora2-regional",
    "pytorch-r4-lora4-regional",
    "pytorch-r1-lora1-full",
    "pytorch-r4-lora4-repeated",
    "pytorch-r4-lora4-pruned",
    "pytorch-r1-lora4-inactive",
    "sage-r1-lora0",
    "sage-r1-lora1",
)


def default_scaling_manifest(
    *,
    repeats: int = 3,
    artifacts: tuple[PerformanceArtifact, ...] = (),
) -> ScalingPerformanceManifest:
    """Return the authoritative RTX 5090 P10.1 scaling definition."""

    pytorch = PerformanceAttentionBackend.PYTORCH
    sage = PerformanceAttentionBackend.SAGE
    half = PerformanceMaskLayout.HALF_FIRST
    partition = PerformanceMaskLayout.HARD_PARTITION
    profiles = (
        _profile("pytorch-r1-lora0", pytorch, 1, half, 0, _ZERO_WORK, trace=True),
        _profile(
            "pytorch-r1-lora1",
            pytorch,
            1,
            half,
            1,
            _ONE_WORK,
            maximum_overhead_percent=15.0,
            trace=True,
        ),
        _profile(
            "pytorch-r1-lora2-stacked",
            pytorch,
            1,
            half,
            2,
            _TWO_WORK,
            trace=True,
        ),
        _profile(
            "pytorch-r1-lora4-stacked",
            pytorch,
            1,
            half,
            4,
            _FOUR_WORK,
            maximum_overhead_percent=35.0,
            trace=True,
        ),
        _profile(
            "pytorch-r2-lora0",
            pytorch,
            2,
            partition,
            0,
            _ZERO_WORK,
            equality="pytorch-r1-lora0",
        ),
        _profile(
            "pytorch-r4-lora0",
            pytorch,
            4,
            partition,
            0,
            _ZERO_WORK,
            equality="pytorch-r1-lora0",
        ),
        _profile(
            "pytorch-r2-lora2-regional",
            pytorch,
            2,
            partition,
            2,
            _TWO_WORK,
            adapter_layout=PerformanceAdapterLayout.DISTINCT_PER_REGION,
        ),
        _profile(
            "pytorch-r4-lora4-regional",
            pytorch,
            4,
            partition,
            4,
            _FOUR_WORK,
            adapter_layout=PerformanceAdapterLayout.DISTINCT_PER_REGION,
        ),
        _profile(
            "pytorch-r1-lora1-full",
            pytorch,
            1,
            PerformanceMaskLayout.FULL_FIRST,
            1,
            _ONE_WORK,
        ),
        _profile(
            "pytorch-r4-lora4-repeated",
            pytorch,
            4,
            partition,
            4,
            _REPEATED_WORK,
            adapter_layout=PerformanceAdapterLayout.REPEATED_PER_REGION,
            equality="pytorch-r1-lora1-full",
            equality_mode=PerformanceEqualityMode.BF16,
            trace=True,
        ),
        _profile(
            "pytorch-r4-lora4-pruned",
            pytorch,
            4,
            PerformanceMaskLayout.REGION_ZERO_FULL,
            4,
            _ONE_WORK,
            adapter_layout=PerformanceAdapterLayout.DISTINCT_PER_REGION,
            equality="pytorch-r1-lora1-full",
            trace=True,
        ),
        _profile(
            "pytorch-r1-lora4-inactive",
            pytorch,
            1,
            half,
            4,
            _ZERO_WORK,
            adapter_layout=PerformanceAdapterLayout.INACTIVE_STACKED,
            equality="pytorch-r1-lora0",
            equality_mode=PerformanceEqualityMode.BF16,
            trace=True,
        ),
        _profile(
            "sage-r1-lora0",
            sage,
            1,
            half,
            0,
            _ZERO_WORK,
            equality="pytorch-r1-lora0",
            equality_mode=PerformanceEqualityMode.BF16,
            trace=True,
        ),
        _profile(
            "sage-r1-lora1",
            sage,
            1,
            half,
            1,
            _ONE_WORK,
            equality="pytorch-r1-lora1",
            equality_mode=PerformanceEqualityMode.BF16,
            trace=True,
        ),
    )
    return ScalingPerformanceManifest(
        benchmark_id="anima-regional-lora-p10.1-scaling-v1",
        width=1024,
        height=1024,
        latent_channels=16,
        context_tokens=512,
        context_features=1024,
        denoiser_calls=30,
        warmup_calls=2,
        repeats=repeats,
        seed=1_029_384_756,
        artifacts=artifacts,
        profiles=profiles,
    )


def _profile(
    profile_id: str,
    backend: PerformanceAttentionBackend,
    region_count: int,
    mask_layout: PerformanceMaskLayout,
    adapter_count: int,
    work: PerformanceWorkExpectation,
    *,
    adapter_layout: PerformanceAdapterLayout | None = None,
    equality: str | None = None,
    equality_mode: PerformanceEqualityMode | None = None,
    maximum_overhead_percent: float | None = None,
    trace: bool = False,
) -> ScalingPerformanceProfile:
    """Construct one concise immutable position from explicit policy values."""

    layout = adapter_layout
    if layout is None:
        layout = (
            PerformanceAdapterLayout.NONE
            if adapter_count == 0
            else PerformanceAdapterLayout.DISTINCT_STACKED
        )
    comparison = equality_mode
    if comparison is None:
        comparison = (
            PerformanceEqualityMode.NONE
            if equality is None
            else PerformanceEqualityMode.EXACT
        )
    return ScalingPerformanceProfile(
        profile_id,
        backend,
        region_count,
        mask_layout,
        layout,
        adapter_count,
        work,
        equality,
        comparison,
        maximum_overhead_percent,
        trace,
    )
