# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify cold-path history decoding and non-overlapping attribution."""

from __future__ import annotations

from tools.sdxl_regional_lora_performance.cold_path_results import (
    decode_cold_path_timing,
    summarize_cold_path,
)


def test_summary_accounts_top_level_stages_without_double_counting_variants() -> None:
    """Keep nested materialization and shell evidence out of the elapsed sum."""

    cold = decode_cold_path_timing(
        _history(
            completed_at_ns=101_000_000,
            run_id="cold",
            records=[
                _stage("admission_resolution", 1.0),
                _stage(
                    "variant_materialization",
                    10.0,
                    parameter_count=2,
                    parameter_bytes=16,
                ),
                _stage("variant_shell", 2.0),
                _stage(
                    "variant_materialization",
                    11.0,
                    parameter_count=3,
                    parameter_bytes=24,
                ),
                _stage("variant_shell", 3.0),
                _stage("template_preparation", 30.0),
                _stage("model_residency", 40.0),
                _stage("sampling", 20.0),
            ],
        ),
        terminal_node_id="9",
        started_at_ns=1_000_000,
        seed=1,
    )
    warm = decode_cold_path_timing(
        _history(
            completed_at_ns=112_000_000,
            run_id="warm",
            records=[
                _stage("model_residency", 1.0),
                _stage("sampling", 9.0),
            ],
        ),
        terminal_node_id="9",
        started_at_ns=102_000_000,
        seed=2,
    )

    summary = summarize_cold_path(cold, warm)

    assert summary.cold_runtime_ms == 100.0
    assert summary.warm_runtime_ms == 10.0
    assert summary.stage_totals_ms["variant_materialization"] == 21.0
    assert summary.top_level_accounted_ms == 91.0
    assert summary.unattributed_ms == 9.0
    assert summary.materialized_parameter_count == 5
    assert summary.materialized_parameter_bytes == 40


def _history(
    *,
    completed_at_ns: int,
    run_id: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Build one generic Comfy terminal history."""

    return {
        "outputs": {
            "9": {
                "cold_path_diagnostics": [
                    {
                        "run_id": run_id,
                        "completed_at_ns": completed_at_ns,
                        "records": records,
                        "model_call_count": 30,
                        "peak_vram_bytes": 1024,
                    }
                ]
            }
        }
    }


def _stage(stage: str, elapsed_ms: float, **metadata: object) -> dict[str, object]:
    """Build one structured cold-stage record."""

    return {"stage": stage, "elapsed_ms": elapsed_ms, **metadata}
