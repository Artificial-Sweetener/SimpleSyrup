"""Verify isolated VRAM measurement, acceptance, and publication."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from tools.anima_regional_lora_performance import (
    isolated_measurement as measurement_module,
)
from tools.anima_regional_lora_performance.isolated_measurement import (
    IsolatedProfileMeasurement,
    IsolatedVramObservation,
    complete_isolated_observation,
    evaluate_isolated_vram_result,
    write_isolated_vram_result,
)
from tools.anima_regional_lora_performance.matrix_manifest import (
    ScalingPerformanceProfile,
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_profile import (
    PreparedScalingProfile,
)
from tools.anima_regional_lora_performance.measurement import (
    PerformanceSequenceMeasurement,
)
from tools.anima_regional_lora_performance.scaling_work_contract import (
    expected_scaling_work,
)


def test_isolated_measurement_retains_no_output_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read finite output identity and discard the measured GPU tensor value."""

    definition = default_scaling_manifest().profiles[0]
    profile = cast(
        PreparedScalingProfile,
        SimpleNamespace(
            definition=definition,
            runtime=SimpleNamespace(prepared_target_count=0),
            attention_override=None,
        ),
    )
    monkeypatch.setattr(
        measurement_module,
        "current_cuda_allocation",
        lambda device: 1_000,
    )
    monkeypatch.setattr(
        measurement_module,
        "execute_call_sequence",
        lambda *args, **kwargs: PerformanceSequenceMeasurement(
            4.0,
            1_500,
            "a" * 64,
            30,
            torch.ones(1),
        ),
    )
    monkeypatch.setattr(
        measurement_module,
        "read_scaling_work_observation",
        lambda selected, sigmas: expected_scaling_work(definition.work),
    )

    result = measurement_module.measure_isolated_profile(
        profile,
        latent=torch.zeros(1),
        context=torch.zeros(1),
        sample_sigmas=torch.ones(31),
        call_count=30,
    )

    assert result.output_finite
    assert result.resident_after_warmup_bytes == 1_000
    assert "output" not in {field.name for field in fields(result)}


def test_isolated_result_accepts_complete_allocation_and_work_matrix(
    tmp_path: Path,
) -> None:
    """Accept all 14 profiles and publish only immutable scalar evidence."""

    manifest = default_scaling_manifest()
    observations = _observations(manifest.profiles)

    result = evaluate_isolated_vram_result(manifest, observations)

    assert result.passed
    path = tmp_path / "result.json"
    write_isolated_vram_result(
        path,
        manifest=manifest,
        result=result,
        environment={"device": "test"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert len(payload["observations"]) == 14
    assert payload["observations"][0]["incremental_peak_bytes"] == 1_200


def test_isolated_result_rejects_retained_cleanup_allocation() -> None:
    """Fail the profile whose post-cleanup allocation exceeds its preload state."""

    manifest = default_scaling_manifest()
    observations = list(_observations(manifest.profiles))
    observations[4] = replace(observations[4], allocation_after_cleanup_bytes=1)

    result = evaluate_isolated_vram_result(manifest, tuple(observations))

    assert not result.passed
    assert not result.profiles[4].allocation_matches


def test_complete_isolated_observation_calculates_incremental_bytes() -> None:
    """Join cleanup readback without retaining session-owned tensors."""

    definition = default_scaling_manifest().profiles[0]
    measured = IsolatedProfileMeasurement(
        definition.profile_id,
        1.0,
        1_500,
        2_000,
        30,
        "a" * 64,
        True,
        0,
        0,
        0,
        expected_scaling_work(definition.work),
    )

    result = complete_isolated_observation(
        definition,
        measured,
        allocation_before_load_bytes=500,
        allocation_after_cleanup_bytes=400,
    )

    assert result.incremental_resident_bytes == 1_000
    assert result.incremental_peak_bytes == 1_500


def _observations(
    profiles: tuple[ScalingPerformanceProfile, ...],
) -> tuple[IsolatedVramObservation, ...]:
    """Build one complete valid isolated evidence matrix."""

    return tuple(
        IsolatedVramObservation(
            profile.profile_id,
            1.0,
            0,
            1_000,
            1_200,
            0,
            1_000,
            1_200,
            30,
            "a" * 64,
            True,
            profile.work.prepared_cache_entries,
            profile.work.prepared_cache_entries,
            10 if profile.profile_id.startswith("sage-") else 0,
            expected_scaling_work(profile.work),
        )
        for profile in profiles
    )
