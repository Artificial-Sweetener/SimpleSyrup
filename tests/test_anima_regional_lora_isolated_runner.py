"""Verify isolated profile ordering and cleanup coordination."""

from __future__ import annotations

import pytest
import torch

from tools.anima_regional_lora_performance import isolated_runner as runner_module
from tools.anima_regional_lora_performance.isolated_measurement import (
    IsolatedProfileMeasurement,
)
from tools.anima_regional_lora_performance.isolated_suite import (
    ValidatedScalingArtifacts,
)
from tools.anima_regional_lora_performance.matrix_manifest import (
    ScalingPerformanceManifest,
    ScalingPerformanceProfile,
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.scaling_work_contract import (
    expected_scaling_work,
)


def test_isolated_runner_verifies_once_and_cleans_before_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visit every profile once with release before both allocation reads."""

    manifest = default_scaling_manifest()
    artifacts = ValidatedScalingArtifacts(*manifest.artifacts)
    events: list[str] = []
    allocations = iter(value for _ in manifest.profiles for value in (0, 0))
    runner = runner_module.AnimaRegionalLoraIsolatedVramRunner()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def validate(
        selected: ScalingPerformanceManifest,
    ) -> ValidatedScalingArtifacts:
        """Record the one artifact validation pass."""

        assert selected is manifest
        events.append("verify")
        return artifacts

    monkeypatch.setattr(runner_module, "validate_scaling_artifacts", validate)
    monkeypatch.setattr(
        runner_module,
        "stabilize_isolated_cuda_runtime",
        lambda device: events.append("stabilize"),
    )
    monkeypatch.setattr(
        runner_module,
        "release_isolated_device",
        lambda device: events.append("release"),
    )

    def allocation(device: torch.device) -> int:
        """Record readback order and return the next clean allocation."""

        events.append("read")
        return next(allocations)

    monkeypatch.setattr(runner_module, "current_cuda_allocation", allocation)

    def measure(
        selected: ScalingPerformanceManifest,
        definition: ScalingPerformanceProfile,
        selected_artifacts: ValidatedScalingArtifacts,
        *,
        expected_device: torch.device,
    ) -> IsolatedProfileMeasurement:
        """Return CPU-only evidence for the current declared profile."""

        assert selected is manifest
        assert selected_artifacts is artifacts
        events.append(f"measure:{definition.profile_id}")
        return IsolatedProfileMeasurement(
            definition.profile_id,
            1.0,
            1_000,
            1_200,
            30,
            "a" * 64,
            True,
            definition.work.prepared_cache_entries,
            definition.work.prepared_cache_entries,
            10 if definition.profile_id.startswith("sage-") else 0,
            expected_scaling_work(definition.work),
        )

    monkeypatch.setattr(runner, "_measure_profile", measure)
    monkeypatch.setattr(
        runner_module,
        "performance_environment",
        lambda device: {"device": str(device)},
    )

    observations, environment = runner.run(manifest)

    assert events.count("verify") == 1
    assert len(observations) == 14
    assert environment == {"device": "cuda:0"}
    assert events[1:6] == [
        "stabilize",
        "release",
        "read",
        f"measure:{manifest.profiles[0].profile_id}",
        "release",
    ]
    assert events[6] == "read"


def test_isolated_runner_releases_after_measurement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release the selected profile when execution raises before publication."""

    manifest = default_scaling_manifest()
    runner = runner_module.AnimaRegionalLoraIsolatedVramRunner()
    released: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        runner_module,
        "validate_scaling_artifacts",
        lambda selected: ValidatedScalingArtifacts(*selected.artifacts),
    )
    monkeypatch.setattr(
        runner_module,
        "stabilize_isolated_cuda_runtime",
        lambda device: None,
    )
    monkeypatch.setattr(
        runner_module,
        "release_isolated_device",
        lambda device: released.append(True),
    )
    monkeypatch.setattr(runner_module, "current_cuda_allocation", lambda device: 0)
    monkeypatch.setattr(
        runner,
        "_measure_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("measure failed")),
    )

    with pytest.raises(RuntimeError, match="measure failed"):
        runner.run(manifest)

    assert released == [True, True]
