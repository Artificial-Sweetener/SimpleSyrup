# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate and persist complete P5.7 performance evidence."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .manifest import PerformanceManifest


@dataclass(frozen=True, slots=True)
class PerformanceObservation:
    """Retain one measured complete denoiser sequence."""

    profile_id: str
    repeat_index: int
    runtime_ms: float
    peak_vram_bytes: int
    denoiser_calls: int
    output_sha256: str
    prepared_target_count: int


@dataclass(frozen=True, slots=True)
class PerformanceProfileResult:
    """Summarize one profile median and its baseline-relative overhead."""

    profile_id: str
    adapter_count: int
    median_runtime_ms: float
    median_peak_vram_bytes: int
    overhead_percent: float
    maximum_overhead_percent: float
    passed: bool


@dataclass(frozen=True, slots=True)
class PerformanceResult:
    """Retain complete observations and the evaluated zero/one/four gate."""

    observations: tuple[PerformanceObservation, ...]
    profiles: tuple[PerformanceProfileResult, ...]

    @property
    def passed(self) -> bool:
        """Return whether every required profile satisfies its gate."""

        return all(profile.passed for profile in self.profiles)


def evaluate(
    manifest: PerformanceManifest,
    observations: tuple[PerformanceObservation, ...],
) -> PerformanceResult:
    """Validate complete evidence and calculate exact median overheads."""

    expected_pairs = {
        (profile.profile_id, repeat_index)
        for profile in manifest.profiles
        for repeat_index in range(manifest.repeats)
    }
    actual_pairs = {
        (observation.profile_id, observation.repeat_index)
        for observation in observations
    }
    if actual_pairs != expected_pairs or len(observations) != len(expected_pairs):
        raise ValueError("P5.7 observations must cover every profile and repeat once.")
    for observation in observations:
        if observation.denoiser_calls != manifest.denoiser_calls:
            raise ValueError("P5.7 denoiser-call counts must remain identical.")
        if observation.runtime_ms <= 0 or observation.peak_vram_bytes <= 0:
            raise ValueError("P5.7 runtime and peak VRAM must be positive.")
        if len(observation.output_sha256) != 64:
            raise ValueError("P5.7 output hashes must be SHA-256 values.")
    for profile in manifest.profiles:
        profile_hashes = {
            observation.output_sha256
            for observation in observations
            if observation.profile_id == profile.profile_id
        }
        if len(profile_hashes) != 1:
            raise ValueError(
                "P5.7 repeated outputs must be deterministic within each profile."
            )

    runtimes = {
        profile.profile_id: statistics.median(
            observation.runtime_ms
            for observation in observations
            if observation.profile_id == profile.profile_id
        )
        for profile in manifest.profiles
    }
    baseline = runtimes[manifest.profiles[0].profile_id]
    profiles = tuple(
        _profile_result(manifest, profile_index, runtimes, baseline, observations)
        for profile_index in range(len(manifest.profiles))
    )
    return PerformanceResult(observations, profiles)


def write_result(
    path: Path,
    *,
    manifest: PerformanceManifest,
    result: PerformanceResult,
    environment: dict[str, object],
) -> None:
    """Atomically write one self-contained JSON benchmark result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "benchmark_id": manifest.benchmark_id,
        "passed": result.passed,
        "environment": environment,
        "settings": {
            "width": manifest.width,
            "height": manifest.height,
            "denoiser_calls": manifest.denoiser_calls,
            "warmup_calls": manifest.warmup_calls,
            "repeats": manifest.repeats,
            "seed": manifest.seed,
        },
        "artifacts": [
            {
                "role": artifact.role,
                "path": str(artifact.path),
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
            }
            for artifact in manifest.artifacts
        ],
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "adapter_count": profile.adapter_count,
                "median_runtime_ms": profile.median_runtime_ms,
                "median_peak_vram_bytes": profile.median_peak_vram_bytes,
                "overhead_percent": profile.overhead_percent,
                "maximum_overhead_percent": profile.maximum_overhead_percent,
                "passed": profile.passed,
            }
            for profile in result.profiles
        ],
        "observations": [
            {
                "profile_id": observation.profile_id,
                "repeat_index": observation.repeat_index,
                "runtime_ms": observation.runtime_ms,
                "peak_vram_bytes": observation.peak_vram_bytes,
                "denoiser_calls": observation.denoiser_calls,
                "output_sha256": observation.output_sha256,
                "prepared_target_count": observation.prepared_target_count,
            }
            for observation in result.observations
        ],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _profile_result(
    manifest: PerformanceManifest,
    profile_index: int,
    runtimes: dict[str, float],
    baseline: float,
    observations: tuple[PerformanceObservation, ...],
) -> PerformanceProfileResult:
    """Calculate one exact profile result from complete observations."""

    profile = manifest.profiles[profile_index]
    runtime = runtimes[profile.profile_id]
    overhead = 0.0 if profile_index == 0 else (runtime / baseline - 1.0) * 100.0
    peak = int(
        statistics.median(
            observation.peak_vram_bytes
            for observation in observations
            if observation.profile_id == profile.profile_id
        )
    )
    return PerformanceProfileResult(
        profile_id=profile.profile_id,
        adapter_count=profile.adapter_count,
        median_runtime_ms=runtime,
        median_peak_vram_bytes=peak,
        overhead_percent=overhead,
        maximum_overhead_percent=profile.maximum_overhead_percent,
        passed=profile_index == 0 or overhead <= profile.maximum_overhead_percent,
    )
