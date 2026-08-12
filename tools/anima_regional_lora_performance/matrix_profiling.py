"""Capture raw and summarized operators for selected scaling positions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile

from .environment import performance_environment
from .matrix_suite import PreparedScalingSuite
from .measurement import activate_measurement_model, execute_call_sequence
from .operator_capture import (
    PerformanceOperatorObservation,
    decode_operator_observations,
)


@dataclass(frozen=True, slots=True)
class ScalingOperatorSummary:
    """Retain exact requested category counts without replacing raw rows."""

    host_to_device: int
    device_to_device: int
    matrix_multiply: int
    batched_matrix_multiply: int
    gather_or_select: int
    scatter_or_index_copy: int
    fused_accumulation: int
    kernel_launch: int


@dataclass(frozen=True, slots=True)
class ScalingProfileCapture:
    """Retain one selected profile's tensor, cache, trace, and operator evidence."""

    profile_id: str
    adapter_count: int
    region_count: int
    attention_backend: str
    denoiser_calls: int
    runtime_ms_with_profiler: float
    peak_vram_bytes: int
    output_sha256: str
    cache_entries: int
    attention_override_calls: int
    trace_file: str
    summary: ScalingOperatorSummary
    operators: tuple[PerformanceOperatorObservation, ...]


def capture_scaling_profiles(
    suite: PreparedScalingSuite,
    *,
    output_directory: Path,
    call_count: int,
) -> Path:
    """Profile every trace-declared position and atomically publish evidence."""

    if not 1 <= call_count <= suite.manifest.denoiser_calls:
        raise ValueError("Scaling profiling calls must fit the benchmark trajectory.")
    output_directory.mkdir(parents=True, exist_ok=False)
    captures: list[ScalingProfileCapture] = []
    for profile, context in zip(suite.profiles, suite.contexts, strict=True):
        if not profile.definition.capture_operator_trace:
            continue
        runtime = profile.runtime
        activate_measurement_model(runtime.model)
        execute_call_sequence(
            runtime,
            latent=suite.latent,
            context=context,
            sample_sigmas=suite.sample_sigmas,
            call_count=suite.manifest.warmup_calls,
            measured=False,
        )
        cache_entries = runtime.prepared_target_count
        override_before = (
            0
            if profile.attention_override is None
            else profile.attention_override.call_count
        )
        trace_path = output_directory / f"{profile.definition.profile_id}.json"
        with torch_profile(
            activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
            profile_memory=True,
            record_shapes=True,
            with_stack=False,
        ) as profiler:
            measurement = execute_call_sequence(
                runtime,
                latent=suite.latent,
                context=context,
                sample_sigmas=suite.sample_sigmas,
                call_count=call_count,
                measured=True,
            )
        profiler.export_chrome_trace(str(trace_path))
        override_after = (
            0
            if profile.attention_override is None
            else profile.attention_override.call_count
        )
        operators = decode_operator_observations(profiler.key_averages())
        captures.append(
            ScalingProfileCapture(
                profile.definition.profile_id,
                profile.definition.adapter_count,
                profile.definition.region_count,
                profile.definition.attention_backend.value,
                measurement.denoiser_calls,
                measurement.runtime_ms,
                measurement.peak_vram_bytes,
                measurement.output_sha256,
                cache_entries,
                override_after - override_before,
                trace_path.name,
                summarize_operators(operators),
                operators,
            )
        )
    if len(captures) != sum(
        profile.capture_operator_trace for profile in suite.manifest.profiles
    ):
        raise AssertionError("Scaling profiling omitted a declared trace position.")
    result_path = output_directory / "profile.json"
    _write_profile_result(result_path, suite=suite, captures=tuple(captures))
    return result_path


def summarize_operators(
    operators: tuple[PerformanceOperatorObservation, ...],
) -> ScalingOperatorSummary:
    """Count named operator categories while retaining complete source rows."""

    return ScalingOperatorSummary(
        host_to_device=_count(operators, lambda name: "Memcpy HtoD" in name),
        device_to_device=_count(operators, lambda name: "Memcpy DtoD" in name),
        matrix_multiply=_count(
            operators,
            lambda name: name in ("aten::mm", "aten::matmul", "aten::addmm"),
        ),
        batched_matrix_multiply=_count(
            operators,
            lambda name: name == "aten::bmm",
        ),
        gather_or_select=_count(
            operators,
            lambda name: name in ("aten::gather", "aten::index_select", "aten::select"),
        ),
        scatter_or_index_copy=_count(
            operators,
            lambda name: "scatter" in name or name == "aten::index_copy_",
        ),
        fused_accumulation=_count(
            operators,
            lambda name: name == "_fused_active_accumulation_kernel",
        ),
        kernel_launch=_count(
            operators,
            lambda name: (
                name in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx")
            ),
        ),
    )


def _count(
    operators: tuple[PerformanceOperatorObservation, ...],
    predicate: object,
) -> int:
    """Sum event counts selected by one internal name predicate."""

    if not callable(predicate):
        raise TypeError("Scaling operator predicate must be callable.")
    return sum(operator.count for operator in operators if predicate(operator.name))


def _write_profile_result(
    path: Path,
    *,
    suite: PreparedScalingSuite,
    captures: tuple[ScalingProfileCapture, ...],
) -> None:
    """Atomically publish complete raw rows and their category summaries."""

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
