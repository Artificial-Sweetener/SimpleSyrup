# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exclusive benchmark capture of cold-stage diagnostics."""

from __future__ import annotations

import logging

import pytest

from tools.attention_coupling_benchmark.comfy_probe.cold_path_capture import (
    CaptureColdPathDiagnosticsV3,
    ReadColdPathDiagnosticsV3,
)

_LOGGER_NAME = "simple_syrup.runtime.regional_lora.standard_unet_cold_path"
_COMPOSITION_LOGGER_NAME = (
    "simple_syrup.runtime.regional_lora.standard_unet_composition"
)


def test_capture_returns_every_ordered_stage_and_restores_logger() -> None:
    """Retain repeated regional stages without deduplication."""

    logger = logging.getLogger(_LOGGER_NAME)
    original_level = logger.level
    model = object()
    started = CaptureColdPathDiagnosticsV3.execute(model, "cold-1")
    assert started.result[0] is model
    logger.debug(
        "materialized left",
        extra={
            "cold_path_diagnostics": {
                "stage": "variant_materialization",
                "elapsed_ms": 10.0,
            }
        },
    )
    logger.debug(
        "materialized right",
        extra={
            "cold_path_diagnostics": {
                "stage": "variant_materialization",
                "elapsed_ms": 11.0,
            }
        },
    )
    logging.getLogger(_COMPOSITION_LOGGER_NAME).debug(
        "model call",
        extra={"regional_composition": {"stage": "regional"}},
    )

    result = ReadColdPathDiagnosticsV3.execute({}, "cold-1")

    capture = result.ui["cold_path_diagnostics"][0]
    assert capture["record_count"] == 2
    assert capture["model_call_count"] == 1
    assert [record["elapsed_ms"] for record in capture["records"]] == [10.0, 11.0]
    assert logger.level == original_level


def test_capture_is_exclusive_and_missing_read_fails_closed() -> None:
    """Prevent overlapping workflows from contaminating unattributed records."""

    CaptureColdPathDiagnosticsV3.execute(object(), "cold-exclusive")
    with pytest.raises(RuntimeError, match="already active"):
        CaptureColdPathDiagnosticsV3.execute(object(), "cold-other")
    logger = logging.getLogger(_LOGGER_NAME)
    logger.debug(
        "sampling",
        extra={
            "cold_path_diagnostics": {
                "stage": "sampling",
                "elapsed_ms": 1.0,
            }
        },
    )
    ReadColdPathDiagnosticsV3.execute({}, "cold-exclusive")
    with pytest.raises(ValueError, match="was not started"):
        ReadColdPathDiagnosticsV3.execute({}, "cold-exclusive")


def test_capture_nodes_are_benchmark_only() -> None:
    """Keep cold attribution outside the public SimpleSyrup node surface."""

    capture = CaptureColdPathDiagnosticsV3.define_schema()
    read = ReadColdPathDiagnosticsV3.define_schema()
    assert capture.node_id == "SimpleSyrupBenchmark.CaptureColdPathDiagnostics"
    assert read.node_id == "SimpleSyrupBenchmark.ReadColdPathDiagnostics"
    assert capture.is_dev_only is True
    assert read.is_dev_only is True
