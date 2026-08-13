# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate and publish complete P10.1 scaling evidence."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .matrix_manifest import (
    PerformanceAttentionBackend,
    PerformanceEqualityMode,
    ScalingPerformanceManifest,
    ScalingPerformanceProfile,
)
from .matrix_measurement import (
    ScalingPerformanceObservation,
)
from .scaling_work_contract import expected_scaling_work

_BF16_RTOL = 0.016
_BF16_ATOL = 0.0078125
_BASELINE_PROFILE_ID = "pytorch-r1-lora0"


@dataclass(frozen=True, slots=True)
class ScalingProfileResult:
    """Summarize one complete position and every acceptance dimension."""

    profile_id: str
    median_runtime_ms: float
    median_peak_vram_bytes: int
    overhead_percent: float | None
    maximum_overhead_percent: float | None
    cache_entries: int
    attention_override_calls: int
    equality_profile_id: str | None
    maximum_absolute_delta: float
    mean_absolute_delta: float
    deterministic: bool
    cache_stable: bool
    work_matches: bool
    fidelity_matches: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class ScalingPerformanceResult:
    """Retain all observations and evaluated profile results."""

    observations: tuple[ScalingPerformanceObservation, ...]
    profiles: tuple[ScalingProfileResult, ...]

    @property
    def passed(self) -> bool:
        """Return whether every declared scaling position is accepted."""

        return all(profile.passed for profile in self.profiles)


def evaluate_scaling_result(
    manifest: ScalingPerformanceManifest,
    observations: tuple[ScalingPerformanceObservation, ...],
) -> ScalingPerformanceResult:
    """Require complete coordinates and evaluate exact matrix acceptance."""

    _validate_coordinates(manifest, observations)
    by_coordinate = {
        (observation.profile_id, observation.repeat_index): observation
        for observation in observations
    }
    baseline_runtime = _median_runtime(
        _observations_for(observations, _BASELINE_PROFILE_ID)
    )
    results = tuple(
        _evaluate_profile(
            manifest,
            profile,
            observations,
            by_coordinate=by_coordinate,
            baseline_runtime=baseline_runtime,
        )
        for profile in manifest.profiles
    )
    return ScalingPerformanceResult(observations, results)


def write_scaling_result(
    path: Path,
    *,
    manifest: ScalingPerformanceManifest,
    result: ScalingPerformanceResult,
    environment: dict[str, object],
) -> None:
    """Atomically publish JSON evidence without retaining output tensors."""

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
            "bf16_rtol": _BF16_RTOL,
            "bf16_atol": _BF16_ATOL,
        },
        "artifacts": [asdict(artifact) for artifact in manifest.artifacts],
        "profile_definitions": [
            {
                **asdict(profile),
                "attention_backend": profile.attention_backend.value,
                "mask_layout": profile.mask_layout.value,
                "adapter_layout": profile.adapter_layout.value,
                "equality_mode": profile.equality_mode.value,
            }
            for profile in manifest.profiles
        ],
        "profiles": [asdict(profile) for profile in result.profiles],
        "observations": [
            {
                "profile_id": observation.profile_id,
                "repeat_index": observation.repeat_index,
                "runtime_ms": observation.runtime_ms,
                "peak_vram_bytes": observation.peak_vram_bytes,
                "denoiser_calls": observation.denoiser_calls,
                "output_sha256": observation.output_sha256,
                "cache_entries_before": observation.cache_entries_before,
                "cache_entries_after": observation.cache_entries_after,
                "attention_override_calls": observation.attention_override_calls,
                "work": asdict(observation.work),
            }
            for observation in result.observations
        ],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_coordinates(
    manifest: ScalingPerformanceManifest,
    observations: tuple[ScalingPerformanceObservation, ...],
) -> None:
    """Reject missing, duplicate, malformed, or nondeterministic evidence."""

    expected = {
        (profile.profile_id, repeat_index)
        for profile in manifest.profiles
        for repeat_index in range(manifest.repeats)
    }
    actual = {
        (observation.profile_id, observation.repeat_index)
        for observation in observations
    }
    if actual != expected or len(observations) != len(expected):
        raise ValueError("P10.1 observations must cover every profile and repeat once.")
    for observation in observations:
        if observation.denoiser_calls != manifest.denoiser_calls:
            raise ValueError("P10.1 denoiser-call counts must remain identical.")
        if observation.runtime_ms <= 0.0 or observation.peak_vram_bytes <= 0:
            raise ValueError("P10.1 runtime and peak VRAM must be positive.")
        if len(observation.output_sha256) != 64:
            raise ValueError("P10.1 output hashes must be SHA-256 values.")
        if (
            observation.output_float32.device.type != "cpu"
            or observation.output_float32.dtype is not torch.float32
        ):
            raise ValueError("P10.1 fidelity tensors must be float32 CPU snapshots.")


def _evaluate_profile(
    manifest: ScalingPerformanceManifest,
    profile: ScalingPerformanceProfile,
    observations: tuple[ScalingPerformanceObservation, ...],
    *,
    by_coordinate: dict[tuple[str, int], ScalingPerformanceObservation],
    baseline_runtime: float,
) -> ScalingProfileResult:
    """Evaluate deterministic, cache, work, fidelity, and timing contracts."""

    selected = _observations_for(observations, profile.profile_id)
    hashes = {observation.output_sha256 for observation in selected}
    tensors = tuple(observation.output_float32 for observation in selected)
    deterministic = len(hashes) == 1 and all(
        torch.equal(tensors[0], tensor) for tensor in tensors[1:]
    )
    cache_stable = all(
        observation.cache_entries_before
        == profile.work.prepared_cache_entries
        == observation.cache_entries_after
        for observation in selected
    )
    expected_work = expected_scaling_work(profile.work)
    work_matches = all(observation.work == expected_work for observation in selected)
    override_calls = sum(
        observation.attention_override_calls for observation in selected
    )
    backend_matches = (
        override_calls > 0
        if profile.attention_backend is PerformanceAttentionBackend.SAGE
        else override_calls == 0
    )
    maximum_delta, mean_delta, fidelity_matches = _fidelity(
        manifest,
        profile,
        selected,
        by_coordinate=by_coordinate,
    )
    runtime = _median_runtime(selected)
    overhead = (
        None
        if profile.maximum_overhead_percent is None
        else (runtime / baseline_runtime - 1.0) * 100.0
    )
    timing_matches = (
        True
        if profile.maximum_overhead_percent is None
        else overhead is not None and overhead <= profile.maximum_overhead_percent
    )
    passed = (
        deterministic
        and cache_stable
        and work_matches
        and backend_matches
        and fidelity_matches
        and timing_matches
    )
    return ScalingProfileResult(
        profile.profile_id,
        runtime,
        int(statistics.median(value.peak_vram_bytes for value in selected)),
        overhead,
        profile.maximum_overhead_percent,
        profile.work.prepared_cache_entries,
        override_calls,
        profile.equality_profile_id,
        maximum_delta,
        mean_delta,
        deterministic,
        cache_stable,
        work_matches,
        fidelity_matches,
        passed,
    )


def _fidelity(
    manifest: ScalingPerformanceManifest,
    profile: ScalingPerformanceProfile,
    observations: tuple[ScalingPerformanceObservation, ...],
    *,
    by_coordinate: dict[tuple[str, int], ScalingPerformanceObservation],
) -> tuple[float, float, bool]:
    """Evaluate exact controls or declared BF16 Sage tolerance."""

    reference_id = profile.equality_profile_id
    if reference_id is None:
        return 0.0, 0.0, True
    differences: list[torch.Tensor] = []
    exact = True
    tolerant = True
    for observation in observations:
        reference = by_coordinate[(reference_id, observation.repeat_index)]
        if observation.output_float32.shape != reference.output_float32.shape:
            return float("inf"), float("inf"), False
        difference = (observation.output_float32 - reference.output_float32).abs()
        differences.append(difference)
        exact = exact and observation.output_sha256 == reference.output_sha256
        exact = exact and torch.equal(
            observation.output_float32,
            reference.output_float32,
        )
        tolerant = tolerant and torch.allclose(
            observation.output_float32,
            reference.output_float32,
            rtol=_BF16_RTOL,
            atol=_BF16_ATOL,
        )
    maximum = max(float(difference.max().item()) for difference in differences)
    mean = statistics.mean(
        float(difference.mean().item()) for difference in differences
    )
    if profile.equality_mode is PerformanceEqualityMode.EXACT:
        matches = exact
    elif profile.equality_mode is PerformanceEqualityMode.BF16:
        matches = tolerant
    else:
        raise ValueError("P10.1 equality reference requires a comparison mode.")
    return maximum, mean, matches


def _observations_for(
    observations: tuple[ScalingPerformanceObservation, ...],
    profile_id: str,
) -> tuple[ScalingPerformanceObservation, ...]:
    """Return one profile's observations in repeat order."""

    return tuple(
        sorted(
            (
                observation
                for observation in observations
                if observation.profile_id == profile_id
            ),
            key=lambda observation: observation.repeat_index,
        )
    )


def _median_runtime(
    observations: tuple[ScalingPerformanceObservation, ...],
) -> float:
    """Return one profile's median measured runtime."""

    return float(statistics.median(value.runtime_ms for value in observations))
