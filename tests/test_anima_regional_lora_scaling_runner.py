# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify interleaved scaling warmup, repeats, and environment coordination."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from tools.anima_regional_lora_performance import matrix_runner as runner_module
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_measurement import (
    ScalingPerformanceObservation,
    ScalingWorkObservation,
)
from tools.anima_regional_lora_performance.matrix_profile import (
    PreparedScalingProfile,
)
from tools.anima_regional_lora_performance.matrix_suite import PreparedScalingSuite
from tools.anima_regional_lora_performance.measurement import (
    PerformanceSequenceMeasurement,
)


def test_scaling_runner_interleaves_complete_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm in order, then visit every profile once per repeat in order."""

    manifest = default_scaling_manifest()
    profiles = tuple(
        cast(
            PreparedScalingProfile,
            SimpleNamespace(
                definition=definition,
                runtime=SimpleNamespace(model=object()),
            ),
        )
        for definition in manifest.profiles
    )
    contexts = tuple(torch.tensor([index]) for index in range(len(profiles)))
    suite = PreparedScalingSuite(
        manifest,
        torch.device("cpu"),
        profiles,
        contexts,
        torch.zeros(1),
        torch.linspace(1.0, 0.0, 31),
    )
    warmups: list[tuple[str, int]] = []
    measurements: list[tuple[str, int]] = []

    @contextmanager
    def session(selected_manifest: object) -> Iterator[PreparedScalingSuite]:
        """Yield the exact prepared suite once."""

        assert selected_manifest is manifest
        yield suite

    def execute_call_sequence(
        runtime: object,
        **values: object,
    ) -> PerformanceSequenceMeasurement:
        """Capture each warmup through its aligned context."""

        profile_index = next(
            index
            for index, profile in enumerate(profiles)
            if profile.runtime is runtime
        )
        assert values["context"] is contexts[profile_index]
        warmups.append(
            (
                profiles[profile_index].definition.profile_id,
                cast(int, values["call_count"]),
            )
        )
        return PerformanceSequenceMeasurement(
            1.0,
            1,
            "a" * 64,
            cast(int, values["call_count"]),
            torch.zeros(1),
        )

    def measure(
        profile: PreparedScalingProfile,
        **values: object,
    ) -> ScalingPerformanceObservation:
        """Capture exact profile/repeat interleaving."""

        repeat_index = cast(int, values["repeat_index"])
        measurements.append((profile.definition.profile_id, repeat_index))
        return ScalingPerformanceObservation(
            profile.definition.profile_id,
            repeat_index,
            1.0,
            1,
            30,
            "b" * 64,
            torch.zeros(1),
            0,
            0,
            0,
            ScalingWorkObservation(0, 0, 0, 0, 0, 0),
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner_module, "scaling_suite_session", session)
    monkeypatch.setattr(runner_module, "activate_measurement_model", lambda model: None)
    monkeypatch.setattr(runner_module, "execute_call_sequence", execute_call_sequence)
    monkeypatch.setattr(runner_module, "measure_scaling_repeat", measure)
    monkeypatch.setattr(
        runner_module,
        "performance_environment",
        lambda device: {"device": str(device)},
    )

    observations, environment = runner_module.AnimaRegionalLoraScalingRunner().run(
        manifest
    )

    assert warmups == [(definition.profile_id, 2) for definition in manifest.profiles]
    assert measurements == [
        (definition.profile_id, repeat_index)
        for repeat_index in range(3)
        for definition in manifest.profiles
    ]
    assert len(observations) == 42
    assert environment == {"device": "cpu"}
