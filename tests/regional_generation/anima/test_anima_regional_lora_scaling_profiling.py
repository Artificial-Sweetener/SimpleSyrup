# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify selected raw scaling traces and operator summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from tools.anima_regional_lora_performance import matrix_profiling as profiling_module
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_profile import (
    PreparedScalingProfile,
)
from tools.anima_regional_lora_performance.matrix_suite import PreparedScalingSuite
from tools.anima_regional_lora_performance.measurement import (
    PerformanceSequenceMeasurement,
)
from tools.anima_regional_lora_performance.operator_capture import (
    PerformanceOperatorObservation,
)


@dataclass(frozen=True, slots=True)
class _Event:
    """Expose the complete profiler aggregate surface."""

    key: str
    count: int
    self_cpu_time_total: float = 1.0
    cpu_time_total: float = 1.0
    self_device_time_total: float = 1.0
    device_time_total: float = 1.0
    self_cpu_memory_usage: int = 0
    cpu_memory_usage: int = 0
    self_device_memory_usage: int = 0
    device_memory_usage: int = 0


class _Profiler:
    """Publish deterministic raw events and trace files."""

    def __enter__(self) -> _Profiler:
        """Return this active profiler."""

        return self

    def __exit__(self, *values: object) -> None:
        """Finish without suppressing failures."""

    def export_chrome_trace(self, path: str) -> None:
        """Materialize one recognizable trace."""

        Path(path).write_text("trace\n", encoding="utf-8")

    def key_averages(self) -> tuple[_Event, ...]:
        """Return one event for every required summary family."""

        return (
            _Event("Memcpy HtoD (Pageable -> Device)", 2),
            _Event("Memcpy DtoD (Device -> Device)", 3),
            _Event("aten::mm", 4),
            _Event("aten::bmm", 5),
            _Event("aten::gather", 6),
            _Event("aten::index_copy_", 7),
            _Event("_fused_active_accumulation_kernel", 8),
            _Event("cudaLaunchKernel", 9),
        )


def test_scaling_profiling_captures_only_declared_positions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Publish nine complete traces with raw rows and exact summaries."""

    manifest = default_scaling_manifest()
    profiles = tuple(
        cast(
            PreparedScalingProfile,
            SimpleNamespace(
                definition=definition,
                runtime=SimpleNamespace(model=object(), prepared_target_count=0),
                attention_override=None,
            ),
        )
        for definition in manifest.profiles
    )
    suite = PreparedScalingSuite(
        manifest,
        torch.device("cpu"),
        profiles,
        tuple(torch.zeros(1) for _profile in profiles),
        torch.zeros(1),
        torch.linspace(1.0, 0.0, 31),
    )
    monkeypatch.setattr(
        profiling_module,
        "activate_measurement_model",
        lambda model: None,
    )
    monkeypatch.setattr(
        profiling_module,
        "execute_call_sequence",
        lambda *args, **kwargs: PerformanceSequenceMeasurement(
            10.0,
            1_000,
            "a" * 64,
            cast(int, kwargs["call_count"]),
            torch.zeros(1),
        ),
    )
    monkeypatch.setattr(profiling_module, "torch_profile", lambda **values: _Profiler())
    monkeypatch.setattr(
        profiling_module,
        "performance_environment",
        lambda device: {"device": str(device)},
    )

    path = profiling_module.capture_scaling_profiles(
        suite,
        output_directory=tmp_path / "capture",
        call_count=1,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["profiles"]) == 9
    assert payload["profiles"][0]["summary"] == {
        "batched_matrix_multiply": 5,
        "device_to_device": 3,
        "fused_accumulation": 8,
        "gather_or_select": 6,
        "host_to_device": 2,
        "kernel_launch": 9,
        "matrix_multiply": 4,
        "scatter_or_index_copy": 7,
    }
    assert all(
        (tmp_path / "capture" / profile["trace_file"]).is_file()
        for profile in payload["profiles"]
    )
    assert all(len(profile["operators"]) == 8 for profile in payload["profiles"])


def test_operator_summary_retains_category_counts() -> None:
    """Count supplied raw rows without changing or filtering them."""

    operator = PerformanceOperatorObservation(
        "aten::index_select",
        12,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
    )

    summary = profiling_module.summarize_operators((operator,))

    assert summary.gather_or_select == 12
    assert summary.kernel_launch == 0
