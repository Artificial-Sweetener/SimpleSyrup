# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify one-profile isolated fixture assembly and cleanup ownership."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import comfy.model_management
import pytest
import torch

from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from tools.anima_regional_lora_performance import isolated_suite as suite_module
from tools.anima_regional_lora_performance.isolated_suite import (
    PreparedIsolatedScalingProfile,
)
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_profile import (
    PreparedScalingProfile,
)
from tools.anima_regional_lora_performance.model_fixture import (
    AnimaPerformanceModelFixture,
)


def test_isolated_artifacts_are_verified_once_before_profile_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify each declared artifact once and retain its exact manifest object."""

    manifest = default_scaling_manifest()
    verified: list[str] = []
    monkeypatch.setattr(
        suite_module,
        "verify_artifact",
        lambda artifact: verified.append(artifact.role),
    )

    result = suite_module.validate_scaling_artifacts(manifest)

    assert verified == ["anima_base", "regional_lora"]
    assert result.model is manifest.artifacts[0]
    assert result.lora is manifest.artifacts[1]


def test_isolated_profile_prepares_only_the_selected_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load, derive, and build inputs for exactly one declared profile."""

    manifest = default_scaling_manifest()
    definition = manifest.profiles[3]
    artifacts = suite_module.ValidatedScalingArtifacts(*manifest.artifacts)
    source = SimpleNamespace(load_device="cpu")
    fixture = cast(
        AnimaPerformanceModelFixture,
        SimpleNamespace(source=source, surface=object(), admission=object()),
    )
    attention = cast(
        AnimaRegionalAttentionExecution,
        SimpleNamespace(contexts=SimpleNamespace(base_context=torch.ones(1))),
    )
    profile = cast(
        PreparedScalingProfile,
        SimpleNamespace(definition=definition),
    )
    built: list[str] = []
    monkeypatch.setattr(suite_module, "load_model_fixture", lambda *args: fixture)
    monkeypatch.setattr(suite_module, "activate_measurement_model", lambda model: None)
    monkeypatch.setattr(
        suite_module,
        "build_matrix_attention_fixture",
        lambda selected, declared, *, device: attention,
    )

    def build_profile(declared: Any, **values: object) -> PreparedScalingProfile:
        """Capture the sole selected declaration and its fixture collaborators."""

        assert values["source"] is source
        assert values["surface"] is fixture.surface
        assert values["admission"] is fixture.admission
        built.append(declared.profile_id)
        return profile

    monkeypatch.setattr(suite_module, "build_scaling_runtime_profile", build_profile)
    monkeypatch.setattr(
        suite_module,
        "build_matrix_input_fixture",
        lambda selected, current, *, device: (
            torch.zeros(1),
            attention.contexts.base_context,
            torch.ones(31),
        ),
    )

    result = suite_module.prepare_isolated_profile(
        manifest,
        definition,
        artifacts,
    )

    assert isinstance(result, PreparedIsolatedScalingProfile)
    assert built == [definition.profile_id]
    assert result.definition is definition
    assert result.profile is profile


def test_isolated_session_unloads_when_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request installed model cleanup even when the selected load fails."""

    manifest = default_scaling_manifest()
    artifacts = suite_module.ValidatedScalingArtifacts(*manifest.artifacts)
    unloaded: list[bool] = []
    monkeypatch.setattr(
        suite_module,
        "prepare_isolated_profile",
        lambda *args: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    monkeypatch.setattr(
        comfy.model_management,
        "unload_all_models",
        lambda: unloaded.append(True),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        with suite_module.isolated_profile_session(
            manifest,
            manifest.profiles[0],
            artifacts,
        ):
            pytest.fail("A failed isolated preparation must not yield.")

    assert unloaded == [True]


def test_cuda_runtime_stabilization_primes_math_then_releases_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Establish the fixed CUDA workspace before any profile allocation read."""

    device = torch.device("cuda", 0)
    tensors = iter((object(), object()))
    allocations: list[tuple[tuple[int, int], torch.device, torch.dtype]] = []
    matrix_inputs: list[tuple[object, object]] = []
    releases: list[torch.device] = []

    def ones(
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object:
        """Record both fixed BF16 workspace operands."""

        allocations.append((shape, device, dtype))
        return next(tensors)

    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(
        torch,
        "mm",
        lambda left, right: matrix_inputs.append((left, right)),
    )
    monkeypatch.setattr(
        suite_module,
        "release_isolated_device",
        lambda selected: releases.append(selected),
    )

    suite_module.stabilize_isolated_cuda_runtime(device)

    assert allocations == [
        ((16, 16), device, torch.bfloat16),
        ((16, 16), device, torch.bfloat16),
    ]
    assert len(matrix_inputs) == 1
    assert releases == [device]
