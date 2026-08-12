"""Verify complete scaling-suite assembly, alignment, and cleanup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import comfy.model_management
import pytest
import torch

from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from tools.anima_regional_lora_performance import matrix_suite as suite_module
from tools.anima_regional_lora_performance.matrix_manifest import (
    ScalingPerformanceManifest,
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_profile import (
    PreparedScalingProfile,
)
from tools.anima_regional_lora_performance.model_fixture import (
    AnimaPerformanceModelFixture,
)


def test_scaling_suite_derives_every_profile_with_aligned_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinate existing artifact, fixture, runtime, and tensor owners once."""

    manifest = default_scaling_manifest()
    source = SimpleNamespace(load_device="cpu")
    fixture = cast(
        AnimaPerformanceModelFixture,
        SimpleNamespace(source=source, surface=object(), admission=object()),
    )
    attentions = {
        definition.profile_id: cast(
            AnimaRegionalAttentionExecution,
            SimpleNamespace(
                contexts=SimpleNamespace(base_context=torch.full((1,), float(index)))
            ),
        )
        for index, definition in enumerate(manifest.profiles)
    }
    prepared = {
        definition.profile_id: cast(
            PreparedScalingProfile,
            SimpleNamespace(definition=definition),
        )
        for definition in manifest.profiles
    }
    verified: list[str] = []
    built: list[str] = []

    monkeypatch.setattr(
        suite_module,
        "verify_artifact",
        lambda artifact: verified.append(artifact.role),
    )
    monkeypatch.setattr(
        suite_module,
        "load_model_fixture",
        lambda model, lora: fixture,
    )
    monkeypatch.setattr(suite_module, "activate_measurement_model", lambda model: None)

    def build_attention(
        selected_manifest: ScalingPerformanceManifest,
        definition: object,
        *,
        device: torch.device,
    ) -> AnimaRegionalAttentionExecution:
        """Return one profile-specific context owner in manifest order."""

        assert selected_manifest is manifest
        assert device == torch.device("cpu")
        return attentions[cast(Any, definition).profile_id]

    def build_profile(definition: Any, **values: object) -> PreparedScalingProfile:
        """Capture one derivation and its shared installed fixture values."""

        assert values["source"] is source
        assert values["surface"] is fixture.surface
        assert values["admission"] is fixture.admission
        assert values["attention"] is attentions[definition.profile_id]
        built.append(definition.profile_id)
        return prepared[definition.profile_id]

    latent = torch.zeros((2, 1))
    sigmas = torch.linspace(1.0, 0.0, 31)

    def build_inputs(
        selected_manifest: ScalingPerformanceManifest,
        attention: AnimaRegionalAttentionExecution,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return shared tensors after all profile derivations."""

        assert selected_manifest is manifest
        assert attention is attentions[manifest.profiles[0].profile_id]
        assert device == torch.device("cpu")
        return latent, torch.zeros(1), sigmas

    monkeypatch.setattr(suite_module, "build_matrix_attention_fixture", build_attention)
    monkeypatch.setattr(suite_module, "build_scaling_runtime_profile", build_profile)
    monkeypatch.setattr(suite_module, "build_matrix_input_fixture", build_inputs)

    result = suite_module.prepare_scaling_suite(manifest)

    assert verified == ["anima_base", "regional_lora"]
    assert built == [definition.profile_id for definition in manifest.profiles]
    assert result.profiles == tuple(prepared[profile_id] for profile_id in built)
    assert result.contexts == tuple(
        attentions[profile_id].contexts.base_context for profile_id in built
    )
    assert result.latent is latent
    assert result.sample_sigmas is sigmas


def test_scaling_suite_session_unloads_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean installed state when any matrix position fails preparation."""

    unloaded: list[bool] = []
    monkeypatch.setattr(
        suite_module,
        "prepare_scaling_suite",
        lambda manifest: (_ for _ in ()).throw(RuntimeError("profile failed")),
    )
    monkeypatch.setattr(
        comfy.model_management,
        "unload_all_models",
        lambda: unloaded.append(True),
    )

    with pytest.raises(RuntimeError, match="profile failed"):
        with suite_module.scaling_suite_session(default_scaling_manifest()):
            pytest.fail("A failed scaling preparation must not yield.")

    assert unloaded == [True]
