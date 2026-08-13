# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure one scaling repeat with production cache and work diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simple_syrup.runtime.regional_lora.anima_lora_diagnostics import (
    AnimaLoraDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleSession,
)

from .matrix_profile import PreparedScalingProfile
from .measurement import execute_call_sequence


@dataclass(frozen=True, slots=True)
class ScalingWorkObservation:
    """Retain exact schedule-specific production diagnostic cardinalities."""

    active_adapter_uses: int
    active_target_count: int
    target_use_count: int
    deduplicated_target_group_count: int
    compatible_projection_batch_count: int
    deduplicated_target_uses: int


@dataclass(frozen=True, slots=True)
class ScalingPerformanceObservation:
    """Retain one repeat's runtime, fidelity, cache, backend, and work evidence."""

    profile_id: str
    repeat_index: int
    runtime_ms: float
    peak_vram_bytes: int
    denoiser_calls: int
    output_sha256: str
    output_float32: torch.Tensor
    cache_entries_before: int
    cache_entries_after: int
    attention_override_calls: int
    work: ScalingWorkObservation


def measure_scaling_repeat(
    profile: PreparedScalingProfile,
    *,
    repeat_index: int,
    latent: torch.Tensor,
    context: torch.Tensor,
    sample_sigmas: torch.Tensor,
    call_count: int,
) -> ScalingPerformanceObservation:
    """Execute one warm-cache repeat and publish authoritative readback."""

    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
        raise TypeError("Scaling repeat index must be an integer.")
    if repeat_index < 0:
        raise ValueError("Scaling repeat index must be non-negative.")
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
    cache_after = profile.runtime.prepared_target_count
    override_after = (
        0
        if profile.attention_override is None
        else profile.attention_override.call_count
    )
    return ScalingPerformanceObservation(
        profile_id=profile.definition.profile_id,
        repeat_index=repeat_index,
        runtime_ms=measurement.runtime_ms,
        peak_vram_bytes=measurement.peak_vram_bytes,
        denoiser_calls=measurement.denoiser_calls,
        output_sha256=measurement.output_sha256,
        output_float32=measurement.output.detach().float().cpu().contiguous(),
        cache_entries_before=cache_before,
        cache_entries_after=cache_after,
        attention_override_calls=override_after - override_before,
        work=read_scaling_work_observation(profile, sample_sigmas),
    )


def read_scaling_work_observation(
    profile: PreparedScalingProfile,
    sample_sigmas: torch.Tensor,
) -> ScalingWorkObservation:
    """Read one production diagnostic snapshot at the measured start sigma."""

    composition = profile.runtime.composition
    if composition is None:
        return ScalingWorkObservation(0, 0, 0, 0, 0, 0)
    values = sample_sigmas.detach().float().cpu().flatten()
    if values.numel() < 1:
        raise ValueError("Scaling sample sigmas cannot be empty.")
    maximum_sigma = float(values.max().item())
    session = RegionalLoraScheduleSession(
        tuple(execution.adapter_plan for execution in composition.executions),
        maximum_sigma=maximum_sigma,
    )
    resolution = session.resolve(float(values[0].item()))
    work = (
        AnimaLoraDiagnosticsBuilder(
            profile.surface,
            composition,
        )
        .build(resolution)
        .work
    )
    return ScalingWorkObservation(
        work.active_adapter_uses,
        work.active_target_count,
        work.target_use_count,
        work.deduplicated_target_group_count,
        work.compatible_projection_batch_count,
        work.deduplicated_target_uses,
    )
