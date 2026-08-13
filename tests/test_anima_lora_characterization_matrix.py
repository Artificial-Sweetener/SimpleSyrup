# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the complete pinned ADAPTER_A characterization matrix."""

from __future__ import annotations

from tools.anima_lora_characterization.matrix import (
    CAPTURE_SEED,
    MEASUREMENT_SEEDS,
    SCHEDULE,
    profiles,
    runs,
)


def test_profiles_cover_zero_one_two_four_and_schedule_boundaries() -> None:
    """Keep stack scaling and every fixed schedule transition mandatory."""

    configured = profiles()

    assert [profile.profile_id for profile in configured] == [
        "static-0",
        "static-1",
        "static-2",
        "static-4",
        "scheduled-1",
        "scheduled-2",
        "scheduled-4",
    ]
    assert [len(profile.adapters) for profile in configured] == [0, 1, 2, 4, 1, 2, 4]
    for profile in configured:
        assert len({adapter.identity for adapter in profile.adapters}) == len(
            profile.adapters
        )
        assert sum(adapter.strength for adapter in profile.adapters) == 1.0 or not (
            profile.adapters
        )
        for adapter in profile.adapters:
            assert adapter.schedule == (SCHEDULE if profile.mode == "scheduled" else ())


def test_runs_separate_timing_from_output_digest_capture() -> None:
    """Prevent denoiser hashing overhead from contaminating runtime evidence."""

    expanded = runs()

    assert len(expanded) == 28
    assert len({run.artifact_id for run in expanded}) == 28
    for profile in profiles():
        profile_runs = [run for run in expanded if run.profile == profile]
        measurement = [run for run in profile_runs if not run.capture_outputs]
        capture = [run for run in profile_runs if run.capture_outputs]
        assert [run.seed for run in measurement] == list(MEASUREMENT_SEEDS)
        assert len(capture) == 1
        assert capture[0].seed == CAPTURE_SEED
