# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify structured standard-UNet composition phase diagnostics."""

from __future__ import annotations

import json
import logging

from simple_syrup.domain.regional_lora_plan import RegionalLoraScheduleBoundary
from simple_syrup.runtime.attention_coupling.unet_attention_phase import (
    StandardUnetAttentionPhase,
    StandardUnetAttentionStage,
)
from simple_syrup.runtime.regional_lora.standard_unet_composition_diagnostics import (
    StandardUnetCompositionDiagnostic,
    StandardUnetCompositionDiagnosticsEmitter,
)


class _Handler(logging.Handler):
    """Retain emitted records without formatting side effects."""

    def __init__(self) -> None:
        """Initialize an empty record list."""

        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append one emitted record."""

        self.records.append(record)


def test_diagnostics_emit_json_compatible_authored_and_effective_strengths() -> None:
    """Expose stage, ownership, and ordered strength composition explicitly."""

    logger = logging.Logger("tests.standard_unet_composition", level=logging.DEBUG)
    handler = _Handler()
    logger.addHandler(handler)
    emitter = StandardUnetCompositionDiagnosticsEmitter(logger)
    diagnostic = StandardUnetCompositionDiagnostic(
        StandardUnetAttentionPhase(
            StandardUnetAttentionStage.SPECIALIZATION,
            0.25,
            0.25,
        ),
        (1.0, 0.25),
        3.5,
        (
            (
                RegionalLoraScheduleBoundary(0.0, 14.0, 1.0, 0),
                RegionalLoraScheduleBoundary(0.5, 3.0, 0.5, 0),
            ),
            (RegionalLoraScheduleBoundary(0.0, 14.0, 0.25, 1),),
        ),
        ("tile",),
    )

    emitter.emit(diagnostic)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.__dict__["operation"] == (
        "standard_unet_regional_composition.resolve"
    )
    values = record.__dict__["regional_composition"]
    assert values == {
        "stage": "specialization",
        "denoising_progress": 0.25,
        "sampling_sigma": 3.5,
        "schedule_multipliers": [1.0, 0.25],
        "adapter_schedules": [
            [
                {
                    "start_percent": 0.0,
                    "start_sigma": 14.0,
                    "strength_multiplier": 1.0,
                    "guarantee_steps": 0,
                },
                {
                    "start_percent": 0.5,
                    "start_sigma": 3.0,
                    "strength_multiplier": 0.5,
                    "guarantee_steps": 0,
                },
            ],
            [
                {
                    "start_percent": 0.0,
                    "start_sigma": 14.0,
                    "strength_multiplier": 0.25,
                    "guarantee_steps": 1,
                }
            ],
        ],
        "spatial_modes": ["tile"],
    }
    json.dumps(values)
