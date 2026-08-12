"""Verify shared Anima performance-suite assembly and cleanup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import comfy.model_management
import pytest
import torch

from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from tools.anima_regional_lora_performance import suite as suite_module
from tools.anima_regional_lora_performance.manifest import (
    PerformanceArtifact,
    PerformanceManifest,
    PerformanceProfile,
    default_manifest,
)
from tools.anima_regional_lora_performance.model_fixture import (
    AnimaPerformanceModelFixture,
)
from tools.anima_regional_lora_performance.runtime_profile import (
    PerformanceRuntimeProfile,
)


def test_suite_assembles_one_resident_source_and_ordered_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse exact artifact, fixture, activation, profile, and tensor owners."""

    manifest = default_manifest()
    source = SimpleNamespace(load_device="cpu")
    fixture = cast(
        AnimaPerformanceModelFixture,
        SimpleNamespace(source=source, surface=object(), admission=object()),
    )
    attention = cast(AnimaRegionalAttentionExecution, object())
    profiles = {
        definition.profile_id: PerformanceRuntimeProfile(
            definition=definition,
            model=object(),
            model_options={"transformer_options": {}},
            executions=(),
            caches=(),
            composition=None,
        )
        for definition in manifest.profiles
    }
    verified: list[str] = []
    activated: list[object] = []
    built: list[str] = []

    def verify_artifact(artifact: PerformanceArtifact) -> None:
        """Capture every manifest artifact before model loading."""

        verified.append(artifact.role)

    def load_model_fixture(
        model_artifact: PerformanceArtifact,
        lora_artifact: PerformanceArtifact,
    ) -> AnimaPerformanceModelFixture:
        """Return one installed-model-shaped fixture after role resolution."""

        assert model_artifact.role == "anima_base"
        assert lora_artifact.role == "regional_lora"
        return fixture

    def activate_measurement_model(model: object) -> None:
        """Capture source activation before any profile construction."""

        activated.append(model)

    def build_attention_fixture(
        selected_manifest: PerformanceManifest,
        *,
        device: torch.device,
    ) -> AnimaRegionalAttentionExecution:
        """Return one shared typed attention authority."""

        assert selected_manifest is manifest
        assert device == torch.device("cpu")
        return attention

    def build_runtime_profile(
        definition: PerformanceProfile,
        **values: object,
    ) -> PerformanceRuntimeProfile:
        """Capture exact profile order and shared construction values."""

        assert values["source"] is source
        assert values["surface"] is fixture.surface
        assert values["attention"] is attention
        assert values["admission"] is fixture.admission
        built.append(definition.profile_id)
        return profiles[definition.profile_id]

    latent = torch.zeros((2, 1))
    context = torch.zeros((2, 1, 1))
    sigmas = torch.tensor([1.0, 0.0])

    def build_input_fixture(
        selected_manifest: PerformanceManifest,
        selected_attention: AnimaRegionalAttentionExecution,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return deterministic tensors through the existing fixture boundary."""

        assert selected_manifest is manifest
        assert selected_attention is attention
        assert device == torch.device("cpu")
        return latent, context, sigmas

    monkeypatch.setattr(suite_module, "verify_artifact", verify_artifact)
    monkeypatch.setattr(suite_module, "load_model_fixture", load_model_fixture)
    monkeypatch.setattr(
        suite_module,
        "activate_measurement_model",
        activate_measurement_model,
    )
    monkeypatch.setattr(
        suite_module,
        "build_attention_fixture",
        build_attention_fixture,
    )
    monkeypatch.setattr(suite_module, "build_runtime_profile", build_runtime_profile)
    monkeypatch.setattr(suite_module, "build_input_fixture", build_input_fixture)

    result = suite_module.prepare_performance_suite(manifest)

    assert verified == ["anima_base", "regional_lora"]
    assert activated == [source]
    assert built == [profile.profile_id for profile in manifest.profiles]
    assert result.profiles == tuple(profiles[profile_id] for profile_id in built)
    assert result.latent is latent
    assert result.context is context
    assert result.sample_sigmas is sigmas


def test_suite_session_unloads_models_when_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean installed model state even when suite assembly raises."""

    unloaded: list[bool] = []

    def fail_preparation(manifest: PerformanceManifest) -> object:
        """Raise after accepting the manifest like a failed host boundary."""

        assert manifest is selected_manifest
        raise RuntimeError("fixture failed")

    def unload_all_models() -> None:
        """Record failure-safe cleanup."""

        unloaded.append(True)

    selected_manifest = default_manifest()
    monkeypatch.setattr(suite_module, "prepare_performance_suite", fail_preparation)
    monkeypatch.setattr(
        comfy.model_management,
        "unload_all_models",
        unload_all_models,
    )

    with pytest.raises(RuntimeError, match="fixture failed"):
        with suite_module.performance_suite_session(selected_manifest):
            pytest.fail("A failed preparation must not yield a suite.")

    assert unloaded == [True]
