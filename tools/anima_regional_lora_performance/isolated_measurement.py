# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure, validate, and publish isolated P10.1 VRAM evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .matrix_manifest import (
    PerformanceAttentionBackend,
    ScalingPerformanceManifest,
    ScalingPerformanceProfile,
)
from .matrix_measurement import (
    ScalingWorkObservation,
    read_scaling_work_observation,
)
from .matrix_profile import PreparedScalingProfile
from .measurement import execute_call_sequence
from .scaling_work_contract import expected_scaling_work


@dataclass(frozen=True, slots=True)
class IsolatedProfileMeasurement:
    """Retain one in-session measurement without preserving GPU tensors."""

    profile_id: str
    runtime_ms: float
    resident_after_warmup_bytes: int
    peak_vram_bytes: int
    denoiser_calls: int
    output_sha256: str
    output_finite: bool
    cache_entries_before: int
    cache_entries_after: int
    attention_override_calls: int
    work: ScalingWorkObservation


@dataclass(frozen=True, slots=True)
class IsolatedVramObservation:
    """Retain one profile's complete allocation and execution evidence."""

    profile_id: str
    runtime_ms: float
    allocation_before_load_bytes: int
    resident_after_warmup_bytes: int
    peak_vram_bytes: int
    allocation_after_cleanup_bytes: int
    incremental_resident_bytes: int
    incremental_peak_bytes: int
    denoiser_calls: int
    output_sha256: str
    output_finite: bool
    cache_entries_before: int
    cache_entries_after: int
    attention_override_calls: int
    work: ScalingWorkObservation


@dataclass(frozen=True, slots=True)
class IsolatedVramProfileResult:
    """Summarize every isolated acceptance dimension for one profile."""

    profile_id: str
    call_count_matches: bool
    output_valid: bool
    cache_matches: bool
    work_matches: bool
    backend_matches: bool
    allocation_matches: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class IsolatedVramResult:
    """Retain the complete ordered observation and acceptance matrix."""

    observations: tuple[IsolatedVramObservation, ...]
    profiles: tuple[IsolatedVramProfileResult, ...]

    @property
    def passed(self) -> bool:
        """Return whether every isolated profile satisfies its contract."""

        return all(profile.passed for profile in self.profiles)


def current_cuda_allocation(device: torch.device) -> int:
    """Read the current allocated CUDA bytes for one exact device."""

    value = int(torch.cuda.memory_allocated(device))
    if value < 0:
        raise ValueError("CUDA allocated memory cannot be negative.")
    return value


def measure_isolated_profile(
    profile: PreparedScalingProfile,
    *,
    latent: torch.Tensor,
    context: torch.Tensor,
    sample_sigmas: torch.Tensor,
    call_count: int,
) -> IsolatedProfileMeasurement:
    """Execute one warm resident profile and discard its GPU output tensor."""

    resident = current_cuda_allocation(latent.device)
    cache_before = profile.runtime.prepared_target_count
    override_before = (
        0
        if profile.attention_override is None
        else profile.attention_override.call_count
    )
    measurement = execute_call_sequence(
        profile.runtime,
        latent=latent,
        context=context,
        sample_sigmas=sample_sigmas,
        call_count=call_count,
        measured=True,
    )
    output_finite = bool(torch.isfinite(measurement.output).all().item())
    cache_after = profile.runtime.prepared_target_count
    override_after = (
        0
        if profile.attention_override is None
        else profile.attention_override.call_count
    )
    result = IsolatedProfileMeasurement(
        profile_id=profile.definition.profile_id,
        runtime_ms=measurement.runtime_ms,
        resident_after_warmup_bytes=resident,
        peak_vram_bytes=measurement.peak_vram_bytes,
        denoiser_calls=measurement.denoiser_calls,
        output_sha256=measurement.output_sha256,
        output_finite=output_finite,
        cache_entries_before=cache_before,
        cache_entries_after=cache_after,
        attention_override_calls=override_after - override_before,
        work=read_scaling_work_observation(profile, sample_sigmas),
    )
    del measurement
    return result


def complete_isolated_observation(
    definition: ScalingPerformanceProfile,
    measurement: IsolatedProfileMeasurement,
    *,
    allocation_before_load_bytes: int,
    allocation_after_cleanup_bytes: int,
) -> IsolatedVramObservation:
    """Join CPU-only measurement evidence with before/after cleanup readback."""

    if measurement.profile_id != definition.profile_id:
        raise ValueError("Isolated measurement profile identity changed.")
    return IsolatedVramObservation(
        profile_id=definition.profile_id,
        runtime_ms=measurement.runtime_ms,
        allocation_before_load_bytes=allocation_before_load_bytes,
        resident_after_warmup_bytes=measurement.resident_after_warmup_bytes,
        peak_vram_bytes=measurement.peak_vram_bytes,
        allocation_after_cleanup_bytes=allocation_after_cleanup_bytes,
        incremental_resident_bytes=(
            measurement.resident_after_warmup_bytes - allocation_before_load_bytes
        ),
        incremental_peak_bytes=(
            measurement.peak_vram_bytes - allocation_before_load_bytes
        ),
        denoiser_calls=measurement.denoiser_calls,
        output_sha256=measurement.output_sha256,
        output_finite=measurement.output_finite,
        cache_entries_before=measurement.cache_entries_before,
        cache_entries_after=measurement.cache_entries_after,
        attention_override_calls=measurement.attention_override_calls,
        work=measurement.work,
    )


def evaluate_isolated_vram_result(
    manifest: ScalingPerformanceManifest,
    observations: tuple[IsolatedVramObservation, ...],
) -> IsolatedVramResult:
    """Require one complete ordered profile matrix and evaluate its contracts."""

    expected_ids = tuple(profile.profile_id for profile in manifest.profiles)
    actual_ids = tuple(observation.profile_id for observation in observations)
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ValueError(
            "Isolated VRAM evidence must cover every profile once in order."
        )
    results = tuple(
        _evaluate_isolated_profile(manifest, definition, observation)
        for definition, observation in zip(
            manifest.profiles,
            observations,
            strict=True,
        )
    )
    return IsolatedVramResult(observations, results)


def write_isolated_vram_result(
    path: Path,
    *,
    manifest: ScalingPerformanceManifest,
    result: IsolatedVramResult,
    environment: dict[str, object],
) -> None:
    """Atomically publish isolated evidence without retaining tensor contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "benchmark_id": f"{manifest.benchmark_id}-isolated-vram",
        "passed": result.passed,
        "environment": environment,
        "settings": {
            "width": manifest.width,
            "height": manifest.height,
            "denoiser_calls": manifest.denoiser_calls,
            "warmup_calls": manifest.warmup_calls,
            "seed": manifest.seed,
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
        "observations": [asdict(observation) for observation in result.observations],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _evaluate_isolated_profile(
    manifest: ScalingPerformanceManifest,
    definition: ScalingPerformanceProfile,
    observation: IsolatedVramObservation,
) -> IsolatedVramProfileResult:
    """Evaluate call, output, cache, work, backend, and allocation invariants."""

    call_count_matches = observation.denoiser_calls == manifest.denoiser_calls
    output_valid = (
        observation.output_finite
        and observation.runtime_ms > 0.0
        and len(observation.output_sha256) == 64
    )
    cache_matches = (
        observation.cache_entries_before
        == definition.work.prepared_cache_entries
        == observation.cache_entries_after
    )
    work_matches = observation.work == expected_scaling_work(definition.work)
    backend_matches = (
        observation.attention_override_calls > 0
        if definition.attention_backend is PerformanceAttentionBackend.SAGE
        else observation.attention_override_calls == 0
    )
    allocation_matches = (
        observation.allocation_before_load_bytes >= 0
        and observation.resident_after_warmup_bytes > 0
        and observation.peak_vram_bytes >= observation.resident_after_warmup_bytes
        and observation.allocation_after_cleanup_bytes
        <= observation.allocation_before_load_bytes
        and observation.incremental_resident_bytes >= 0
        and observation.incremental_peak_bytes >= observation.incremental_resident_bytes
    )
    passed = (
        call_count_matches
        and output_valid
        and cache_matches
        and work_matches
        and backend_matches
        and allocation_matches
    )
    return IsolatedVramProfileResult(
        definition.profile_id,
        call_count_matches,
        output_valid,
        cache_matches,
        work_matches,
        backend_matches,
        allocation_matches,
        passed,
    )
