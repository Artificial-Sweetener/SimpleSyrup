"""Verify complete Anima operator-profile evidence publication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tools.anima_regional_lora_performance import profiling as profiling_module
from tools.anima_regional_lora_performance.manifest import default_manifest
from tools.anima_regional_lora_performance.measurement import (
    PerformanceSequenceMeasurement,
)
from tools.anima_regional_lora_performance.runtime_profile import (
    PerformanceRuntimeProfile,
)
from tools.anima_regional_lora_performance.suite import PreparedPerformanceSuite


@dataclass(frozen=True, slots=True)
class _Event:
    """Expose one complete PyTorch aggregate event surface."""

    key: str
    count: int
    self_cpu_time_total: float
    cpu_time_total: float
    self_device_time_total: float
    device_time_total: float
    self_cpu_memory_usage: int
    cpu_memory_usage: int
    self_device_memory_usage: int
    device_memory_usage: int


class _Profiler:
    """Publish deterministic events and one trace file per profile."""

    def __enter__(self) -> _Profiler:
        """Return this active fake profiler."""

        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Finish capture without suppressing failures."""

    def export_chrome_trace(self, path: str) -> None:
        """Write one recognizable external trace artifact."""

        Path(path).write_text("trace\n", encoding="utf-8")

    def key_averages(self) -> tuple[_Event, ...]:
        """Return events deliberately outside desired sort order."""

        return (
            _Event("cpu-owner", 2, 20.0, 30.0, 1.0, 2.0, 3, 4, 5, 6),
            _Event("cuda-owner", 1, 10.0, 15.0, 40.0, 50.0, 7, 8, 9, 10),
        )


def test_profiling_publishes_every_profile_operator_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Capture all declared profiles and retain every aggregate event row."""

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
    calls: list[tuple[str, int]] = []

    def execute_call_sequence(
        profile: PerformanceRuntimeProfile,
        *,
        latent: torch.Tensor,
        context: torch.Tensor,
        sample_sigmas: torch.Tensor,
        call_count: int,
        measured: bool,
    ) -> PerformanceSequenceMeasurement:
        """Return deterministic evidence while recording warmup/profile depth."""

        assert latent is suite.latent
        assert context is suite.context
        assert sample_sigmas is suite.sample_sigmas
        assert measured is False
        calls.append((profile.definition.profile_id, call_count))
        return PerformanceSequenceMeasurement(
            10.0,
            1_000,
            str(profile.definition.adapter_count + 1) * 64,
            call_count,
            torch.zeros(1),
        )

    monkeypatch.setattr(
        profiling_module,
        "activate_measurement_model",
        lambda model: None,
    )
    monkeypatch.setattr(
        profiling_module,
        "execute_call_sequence",
        execute_call_sequence,
    )
    monkeypatch.setattr(
        profiling_module,
        "torch_profile",
        lambda **settings: _Profiler(),
    )
    monkeypatch.setattr(
        profiling_module,
        "performance_environment",
        lambda device: {"device": str(device)},
    )

    result_path = profiling_module.capture_performance_profiles(
        suite,
        output_directory=tmp_path / "capture",
        call_count=1,
    )
    payload: object = json.loads(result_path.read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
    assert [profile["profile_id"] for profile in payload["profiles"]] == [
        definition.profile_id for definition in manifest.profiles
    ]
    assert calls == [
        (definition.profile_id, call_count)
        for definition in manifest.profiles
        for call_count in (2, 1)
    ]
    for profile in payload["profiles"]:
        assert [event["name"] for event in profile["operators"]] == [
            "cuda-owner",
            "cpu-owner",
        ]
        assert (tmp_path / "capture" / profile["trace_file"]).is_file()


@pytest.mark.parametrize("call_count", [0, 31])
def test_profiling_rejects_calls_outside_the_fixed_trajectory(
    tmp_path: Path,
    call_count: int,
) -> None:
    """Reject empty or expanded traces before creating an evidence directory."""

    manifest = default_manifest()
    suite = PreparedPerformanceSuite(
        manifest,
        torch.device("cpu"),
        (),
        torch.zeros(1),
        torch.zeros(1),
        torch.linspace(1.0, 0.0, 31),
    )

    with pytest.raises(ValueError, match="must fit"):
        profiling_module.capture_performance_profiles(
            suite,
            output_directory=tmp_path / "not-created",
            call_count=call_count,
        )

    assert not (tmp_path / "not-created").exists()
