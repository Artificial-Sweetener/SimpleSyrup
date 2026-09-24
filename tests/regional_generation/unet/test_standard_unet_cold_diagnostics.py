# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify opt-in standard-UNet cold-stage diagnostics."""

from __future__ import annotations

import logging

import torch

from simple_syrup.runtime.regional_lora.standard_unet_cold_diagnostics import (
    StandardUnetColdPathDiagnosticsEmitter,
    StandardUnetColdStage,
)


def test_enabled_measurement_emits_elapsed_time_and_bounded_metadata() -> None:
    """Publish one structured stage after its measured work completes."""

    logger = logging.getLogger("tests.standard_unet_cold.enabled")
    logger.setLevel(logging.DEBUG)
    ticks = iter((1_000, 4_500))
    synchronizations: list[torch.device | None] = []
    emitter = StandardUnetColdPathDiagnosticsEmitter(
        logger=logger,
        clock_ns=lambda: next(ticks),
        synchronize=synchronizations.append,
    )
    handler = _RecordHandler()
    logger.addHandler(handler)
    try:
        with emitter.measure(
            StandardUnetColdStage.VARIANT_MATERIALIZATION,
            device=torch.device("cpu"),
        ) as metadata:
            metadata["parameter_count"] = 2
            metadata["parameter_bytes"] = 16
    finally:
        logger.removeHandler(handler)

    assert synchronizations == [torch.device("cpu"), torch.device("cpu")]
    assert len(handler.records) == 1
    payload = getattr(handler.records[0], "cold_path_diagnostics", None)
    assert payload == {
        "stage": "variant_materialization",
        "elapsed_ms": 0.0035,
        "parameter_count": 2,
        "parameter_bytes": 16,
    }


def test_disabled_measurement_avoids_clock_and_synchronization() -> None:
    """Keep ordinary execution free of attribution work and device barriers."""

    logger = logging.getLogger("tests.standard_unet_cold.disabled")
    logger.setLevel(logging.INFO)

    def fail_clock() -> int:
        raise AssertionError("disabled diagnostics must not read the clock")

    def fail_synchronize(_device: torch.device | None) -> None:
        raise AssertionError("disabled diagnostics must not synchronize")

    emitter = StandardUnetColdPathDiagnosticsEmitter(
        logger=logger,
        clock_ns=fail_clock,
        synchronize=fail_synchronize,
    )

    with emitter.measure(StandardUnetColdStage.SAMPLING) as metadata:
        metadata["model_call_count"] = 30


class _RecordHandler(logging.Handler):
    """Retain exact records from one focused emitter test."""

    def __init__(self) -> None:
        """Create an empty record list."""

        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append one emitted diagnostic record."""

        self.records.append(record)
