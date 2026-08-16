# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify strict SDXL image-free diagnostic history decoding."""

from __future__ import annotations

import pytest

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_diagnostic_history import (
    decode_sdxl_visual_diagnostic_history,
)


def test_decoder_returns_exact_metrics_and_diagnostics() -> None:
    """Narrow one complete pair of managed terminal objects."""

    metrics: JsonObject = {"model_call_count": 30, "runtime_ms": 10.0}
    diagnostics: JsonObject = {
        "attention_phase_record_count": 30,
        "self_attention_record_count": 0,
    }
    observed = decode_sdxl_visual_diagnostic_history(
        _history(metrics, diagnostics),
        metrics_node_id="24",
        diagnostics_node_id="25",
    )

    assert observed.metrics is metrics
    assert observed.diagnostics is diagnostics


@pytest.mark.parametrize(
    ("history", "match"),
    (
        ({}, "missing outputs"),
        ({"outputs": {}}, "missing node 24"),
        (
            {
                "outputs": {
                    "24": {"benchmark_metrics": []},
                    "25": {"regional_diagnostics": [{}]},
                }
            },
            "invalid benchmark_metrics",
        ),
    ),
)
def test_decoder_rejects_incomplete_terminal_evidence(
    history: JsonObject,
    match: str,
) -> None:
    """Fail closed when managed history cannot prove both terminals."""

    with pytest.raises(ValueError, match=match):
        decode_sdxl_visual_diagnostic_history(
            history,
            metrics_node_id="24",
            diagnostics_node_id="25",
        )


def _history(metrics: JsonObject, diagnostics: JsonObject) -> JsonObject:
    """Return one complete generic Comfy history value."""

    return {
        "outputs": {
            "24": {"benchmark_metrics": [metrics]},
            "25": {"regional_diagnostics": [diagnostics]},
        }
    }
