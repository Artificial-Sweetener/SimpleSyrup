"""Capture complete PyTorch operator evidence for Anima performance profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile

from .environment import performance_environment
from .measurement import activate_measurement_model, execute_call_sequence
from .operator_capture import (
    PerformanceOperatorObservation,
    decode_operator_observations,
)
from .suite import PreparedPerformanceSuite


@dataclass(frozen=True, slots=True)
class PerformanceProfileCapture:
    """Retain one profile's exact output and complete operator aggregates."""

    profile_id: str
    adapter_count: int
    denoiser_calls: int
    runtime_ms_with_profiler: float
    peak_vram_bytes: int
    output_sha256: str
    prepared_target_count: int
    trace_file: str
    operators: tuple[PerformanceOperatorObservation, ...]


def capture_performance_profiles(
    suite: PreparedPerformanceSuite,
    *,
    output_directory: Path,
    call_count: int,
) -> Path:
    """Profile every declared profile and atomically publish complete evidence."""

    if not 1 <= call_count <= suite.manifest.denoiser_calls:
        raise ValueError(
            "Anima performance profiling calls must fit the benchmark trajectory."
        )
    output_directory.mkdir(parents=True, exist_ok=False)
    captures: list[PerformanceProfileCapture] = []
    for runtime_profile in suite.profiles:
        activate_measurement_model(runtime_profile.model)
        execute_call_sequence(
            runtime_profile,
            latent=suite.latent,
            context=suite.context,
            sample_sigmas=suite.sample_sigmas,
            call_count=suite.manifest.warmup_calls,
            measured=False,
        )
        trace_path = output_directory / f"{runtime_profile.definition.profile_id}.json"
        with torch_profile(
            activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
            profile_memory=True,
            record_shapes=True,
            with_stack=False,
        ) as profiler:
            measurement = execute_call_sequence(
                runtime_profile,
                latent=suite.latent,
                context=suite.context,
                sample_sigmas=suite.sample_sigmas,
                call_count=call_count,
                measured=False,
            )
        profiler.export_chrome_trace(str(trace_path))
        captures.append(
            PerformanceProfileCapture(
                profile_id=runtime_profile.definition.profile_id,
                adapter_count=runtime_profile.definition.adapter_count,
                denoiser_calls=measurement.denoiser_calls,
                runtime_ms_with_profiler=measurement.runtime_ms,
                peak_vram_bytes=measurement.peak_vram_bytes,
                output_sha256=measurement.output_sha256,
                prepared_target_count=runtime_profile.prepared_target_count,
                trace_file=trace_path.name,
                operators=decode_operator_observations(profiler.key_averages()),
            )
        )
    result_path = output_directory / "profile.json"
    _write_result(result_path, suite=suite, captures=tuple(captures))
    return result_path


def _write_result(
    path: Path,
    *,
    suite: PreparedPerformanceSuite,
    captures: tuple[PerformanceProfileCapture, ...],
) -> None:
    """Atomically publish one self-contained profiling summary."""

    payload = {
        "schema_version": 1,
        "benchmark_id": suite.manifest.benchmark_id,
        "environment": performance_environment(suite.device),
        "settings": {
            "width": suite.manifest.width,
            "height": suite.manifest.height,
            "warmup_calls": suite.manifest.warmup_calls,
            "profiled_calls": captures[0].denoiser_calls if captures else 0,
        },
        "profiles": [asdict(capture) for capture in captures],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
