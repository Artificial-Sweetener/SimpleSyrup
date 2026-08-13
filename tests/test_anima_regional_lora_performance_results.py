# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove complete P5.7 median overhead and denoiser-count evaluation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.anima_regional_lora_performance.manifest import default_manifest
from tools.anima_regional_lora_performance.results import (
    PerformanceObservation,
    evaluate,
    write_result,
)


def test_results_pass_exact_one_and_four_adapter_limits(tmp_path: Path) -> None:
    """Calculate medians and persist complete passing evidence."""

    manifest = default_manifest()
    observations = _observations((100.0, 110.0, 130.0))

    result = evaluate(manifest, observations)
    path = tmp_path / "result.json"
    write_result(path, manifest=manifest, result=result, environment={"gpu": "test"})
    payload: object = json.loads(path.read_text(encoding="utf-8"))

    assert result.passed is True
    assert [profile.overhead_percent for profile in result.profiles] == [
        0.0,
        pytest.approx(10.0),
        pytest.approx(30.0),
    ]
    assert isinstance(payload, dict)
    assert payload["passed"] is True
    assert len(payload["observations"]) == 9


def test_results_fail_overhead_without_reducing_the_gate() -> None:
    """Report a measured miss as failed rather than rounding or deferring it."""

    result = evaluate(default_manifest(), _observations((100.0, 116.0, 136.0)))

    assert result.passed is False
    assert [profile.passed for profile in result.profiles] == [True, False, False]


def test_results_reject_missing_duplicate_or_changed_call_counts() -> None:
    """Require every repeat and the identical complete denoiser trajectory."""

    observations = _observations((100.0, 110.0, 130.0))
    with pytest.raises(ValueError, match="every profile and repeat"):
        evaluate(default_manifest(), observations[:-1])
    with pytest.raises(ValueError, match="denoiser-call counts"):
        evaluate(
            default_manifest(),
            (replace(observations[0], denoiser_calls=29), *observations[1:]),
        )


def test_results_reject_output_drift_between_identical_repeats() -> None:
    """Make deterministic denoiser output part of the performance gate."""

    observations = _observations((100.0, 110.0, 130.0))
    changed = replace(observations[1], output_sha256="f" * 64)

    with pytest.raises(ValueError, match="deterministic within each profile"):
        evaluate(default_manifest(), (observations[0], changed, *observations[2:]))


def _observations(
    runtimes: tuple[float, float, float],
) -> tuple[PerformanceObservation, ...]:
    """Build three stable repeats for each required profile."""

    profiles = default_manifest().profiles
    return tuple(
        PerformanceObservation(
            profile_id=profile.profile_id,
            repeat_index=repeat_index,
            runtime_ms=runtimes[profile_index] + (repeat_index - 1),
            peak_vram_bytes=1_000 + profile_index,
            denoiser_calls=30,
            output_sha256=hex(profile_index + 10)[2:] * 64,
            prepared_target_count=profile.adapter_count * 448,
        )
        for profile_index, profile in enumerate(profiles)
        for repeat_index in range(3)
    )
