# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the thin Anima performance matrix coordinator."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

from tools.anima_regional_lora_performance import runner as runner_module
from tools.anima_regional_lora_performance.manifest import default_manifest
from tools.anima_regional_lora_performance.measurement import (
    PerformanceSequenceMeasurement,
)
from tools.anima_regional_lora_performance.runtime_profile import (
    PerformanceRuntimeProfile,
)
from tools.anima_regional_lora_performance.suite import PreparedPerformanceSuite


def test_runner_consumes_one_suite_for_warmup_repeats_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinate measurements without reproducing suite or result policy."""

    manifest = default_manifest()
    profiles = tuple(
        PerformanceRuntimeProfile(
            definition=definition,
            model=object(),
            model_options={"transformer_options": {}},
            executions=(),
            caches=(),
            composition=None,
        )
        for definition in manifest.profiles
    )
    suite = PreparedPerformanceSuite(
        manifest,
        torch.device("cpu"),
        profiles,
        torch.zeros((2, 1)),
        torch.zeros((2, 1, 1)),
        torch.linspace(1.0, 0.0, 31),
    )
    activations: list[object] = []
    calls: list[tuple[str, int, bool]] = []

    @contextmanager
    def performance_suite_session(
        selected_manifest: object,
    ) -> Iterator[PreparedPerformanceSuite]:
        """Yield the exact prepared suite once."""

        assert selected_manifest is manifest
        yield suite

    def activate_measurement_model(model: object) -> None:
        """Capture every warmup and repeat activation."""

        activations.append(model)

    def execute_call_sequence(
        profile: PerformanceRuntimeProfile,
        *,
        latent: torch.Tensor,
        context: torch.Tensor,
        sample_sigmas: torch.Tensor,
        call_count: int,
        measured: bool,
    ) -> PerformanceSequenceMeasurement:
        """Return stable complete evidence for the coordinator."""

        assert latent is suite.latent
        assert context is suite.context
        assert sample_sigmas is suite.sample_sigmas
        calls.append((profile.definition.profile_id, call_count, measured))
        return PerformanceSequenceMeasurement(
            float(call_count),
            1_000,
            str(profile.definition.adapter_count + 1) * 64,
            call_count,
            torch.zeros(1),
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runner_module,
        "performance_suite_session",
        performance_suite_session,
    )
    monkeypatch.setattr(
        runner_module,
        "activate_measurement_model",
        activate_measurement_model,
    )
    monkeypatch.setattr(
        runner_module,
        "execute_call_sequence",
        execute_call_sequence,
    )
    monkeypatch.setattr(
        runner_module,
        "performance_environment",
        lambda device: {"device": str(device)},
    )

    observations, environment = runner_module.AnimaRegionalLoraPerformanceRunner().run(
        manifest
    )

    assert len(observations) == 9
    assert environment == {"device": "cpu"}
    assert calls[:3] == [
        (definition.profile_id, 2, False) for definition in manifest.profiles
    ]
    assert calls[3:] == [
        (definition.profile_id, 30, True)
        for _repeat in range(3)
        for definition in manifest.profiles
    ]
    assert len(activations) == 12
