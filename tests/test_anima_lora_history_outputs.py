# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify strict P0.7 Comfy history decoding."""

from __future__ import annotations

import pytest
from anima_lora_characterization_fixtures import metrics

from tools.anima_lora_characterization.history_outputs import parse_completed_outputs
from tools.anima_lora_characterization.matrix import runs
from tools.comfy_api import JsonObject


def test_history_decoder_returns_one_probe_record_and_image() -> None:
    """Decode the exact output-node fields emitted by real Comfy history."""

    run = runs()[0]
    history: JsonObject = {
        "status": {"status_str": "success"},
        "outputs": {
            "10": {"lora_benchmark_metrics": [metrics(run)]},
            "12": {
                "images": [
                    {"filename": "a.png", "subfolder": "bench", "type": "output"}
                ]
            },
        },
    }

    outputs = parse_completed_outputs(history, metrics_node_id="10", save_node_id="12")

    assert outputs.metrics["run_id"] == run.artifact_id
    assert outputs.image.filename == "a.png"


def test_history_decoder_rejects_failed_or_ambiguous_outputs() -> None:
    """Fail closed on server failure and multiple generated images."""

    with pytest.raises(RuntimeError, match="failed"):
        parse_completed_outputs(
            {"status": {"status_str": "error", "messages": ["failed"]}},
            metrics_node_id="1",
            save_node_id="2",
        )
