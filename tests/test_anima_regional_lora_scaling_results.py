"""Verify complete scaling acceptance and atomic JSON publication."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tools.anima_regional_lora_performance.matrix_manifest import (
    ScalingPerformanceProfile,
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_measurement import (
    ScalingPerformanceObservation,
    ScalingWorkObservation,
)
from tools.anima_regional_lora_performance.matrix_results import (
    evaluate_scaling_result,
    write_scaling_result,
)


def test_scaling_result_accepts_complete_exact_and_sage_controls(
    tmp_path: Path,
) -> None:
    """Evaluate every coordinate, work count, equality, tolerance, and limit."""

    manifest = default_scaling_manifest()
    observations = _observations(manifest.profiles, repeats=manifest.repeats)
    result = evaluate_scaling_result(manifest, observations)

    assert result.passed
    assert all(profile.passed for profile in result.profiles)
    assert result.profiles[1].overhead_percent == pytest.approx(10.0)
    assert result.profiles[3].overhead_percent == pytest.approx(20.0)
    assert result.profiles[9].cache_entries == 448
    assert result.profiles[-1].maximum_absolute_delta == pytest.approx(
        0.001,
        abs=1e-6,
    )
    assert result.profiles[-1].fidelity_matches

    path = tmp_path / "result.json"
    write_scaling_result(
        path,
        manifest=manifest,
        result=result,
        environment={"device": "test"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert len(payload["profiles"]) == 14
    assert len(payload["observations"]) == 42
    assert "output_float32" not in payload["observations"][0]


def test_scaling_result_rejects_cache_growth() -> None:
    """Fail a position when immutable warm-cache cardinality changes."""

    manifest = default_scaling_manifest()
    observations = list(_observations(manifest.profiles, repeats=manifest.repeats))
    observations[1] = replace(observations[1], cache_entries_after=449)

    result = evaluate_scaling_result(manifest, tuple(observations))

    assert not result.passed
    assert not result.profiles[1].cache_stable


def test_scaling_result_rejects_sage_outside_bf16_tolerance() -> None:
    """Keep Sage measured but unsupported when final tensors materially diverge."""

    manifest = default_scaling_manifest()
    observations = list(_observations(manifest.profiles, repeats=manifest.repeats))
    sage_index = next(
        index
        for index, observation in enumerate(observations)
        if observation.profile_id == "sage-r1-lora0"
    )
    observations[sage_index] = replace(
        observations[sage_index],
        output_float32=torch.tensor([2.0]),
    )

    result = evaluate_scaling_result(manifest, tuple(observations))

    assert not result.passed
    assert not result.profiles[-2].fidelity_matches


def _observations(
    profiles: tuple[ScalingPerformanceProfile, ...],
    *,
    repeats: int,
) -> tuple[ScalingPerformanceObservation, ...]:
    """Build complete deterministic evidence with exact equality references."""

    outputs: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    for index, profile in enumerate(profiles):
        if profile.equality_profile_id is None:
            output = torch.tensor([float(index + 1)])
            output_hash = f"{index + 1:x}" * 64
        else:
            reference = outputs[profile.equality_profile_id]
            if profile.profile_id.startswith("sage-"):
                output = reference + 0.001
                output_hash = f"{index + 1:x}" * 64
            else:
                output = reference.clone()
                output_hash = hashes[profile.equality_profile_id]
        outputs[profile.profile_id] = output
        hashes[profile.profile_id] = output_hash[:64]
    values: list[ScalingPerformanceObservation] = []
    for repeat_index in range(repeats):
        for profile in profiles:
            runtime = {
                "pytorch-r1-lora0": 100.0,
                "pytorch-r1-lora1": 110.0,
                "pytorch-r1-lora4-stacked": 120.0,
            }.get(profile.profile_id, 105.0)
            values.append(
                ScalingPerformanceObservation(
                    profile.profile_id,
                    repeat_index,
                    runtime,
                    1_000,
                    30,
                    hashes[profile.profile_id],
                    outputs[profile.profile_id].clone().float(),
                    profile.work.prepared_cache_entries,
                    profile.work.prepared_cache_entries,
                    10 if profile.profile_id.startswith("sage-") else 0,
                    ScalingWorkObservation(
                        profile.work.active_adapter_uses,
                        profile.work.active_target_count,
                        profile.work.target_use_count,
                        profile.work.deduplicated_target_group_count,
                        profile.work.compatible_projection_batch_count,
                        profile.work.deduplicated_target_uses,
                    ),
                )
            )
    return tuple(values)
