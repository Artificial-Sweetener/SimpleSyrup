# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify strict SDXL steady-state metric decoding and summarization."""

import pytest

from tools.comfy_api import JsonObject
from tools.sdxl_regional_lora_performance.measurement import (
    SdxlRegionalLoraTiming,
    decode_timing,
    median_runtime_ms,
)


def test_decodes_one_metric_and_uses_median() -> None:
    """Retain exact call topology while rejecting undersized comparisons."""

    history: JsonObject = {
        "outputs": {
            "metrics": {
                "benchmark_metrics": [
                    {
                        "runtime_ms": 20.0,
                        "model_call_count": 30,
                        "model_input_batch_elements": 60,
                        "model_input_batch_sizes": [2] * 30,
                        "peak_vram_bytes": 100,
                    }
                ]
            }
        }
    }
    decoded = decode_timing(history, "metrics")
    assert decoded.model_call_count == 30
    values = (
        decoded,
        SdxlRegionalLoraTiming(10.0, 30, 60, (2,) * 30, 100),
        SdxlRegionalLoraTiming(30.0, 30, 60, (2,) * 30, 100),
    )
    assert median_runtime_ms(values) == 20.0
    with pytest.raises(ValueError, match="three repeats"):
        median_runtime_ms(values[:2])
