# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the focused benchmark synchronized phase owner."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import torch

from tools.attention_coupling_benchmark.comfy_probe.synchronized_phase_timing import (
    measure_synchronized_phase,
    model_device,
)

_LOGGER_NAME = "simple_syrup.runtime.regional_lora.standard_unet_cold_path"


def test_phase_timing_is_transparent_when_capture_is_disabled(caplog: Any) -> None:
    """Avoid logging or synchronization work outside a diagnostic capture."""

    value = object()
    with measure_synchronized_phase("disabled", device=torch.device("cpu")):
        actual = value

    assert actual is value
    assert not caplog.records


def test_phase_timing_emits_one_structured_record(caplog: Any) -> None:
    """Publish one non-negative elapsed duration under the focused logger."""

    with (
        caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        measure_synchronized_phase("measured", device=torch.device("cpu")),
    ):
        pass

    payloads = [
        record.cold_path_diagnostics
        for record in caplog.records
        if hasattr(record, "cold_path_diagnostics")
    ]
    assert len(payloads) == 1
    assert payloads[0]["stage"] == "measured"
    assert payloads[0]["elapsed_ms"] >= 0.0


def test_model_device_accepts_only_explicit_torch_devices() -> None:
    """Never infer synchronization ownership from arbitrary device-like values."""

    device = torch.device("cpu")
    assert model_device(SimpleNamespace(load_device=device)) is device
    assert model_device(SimpleNamespace(load_device="cuda")) is None
    assert model_device(object()) is None
