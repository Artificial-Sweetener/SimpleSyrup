# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify structured standard-UNet attention-phase diagnostics."""

from __future__ import annotations

import json
import logging

import torch

from simple_syrup.runtime.attention_coupling.unet_attention_phase_diagnostics import (
    StandardUnetAttentionPhaseDiagnosticsEmitter,
)
from simple_syrup.runtime.attention_coupling.unet_attention_phase_session import (
    StandardUnetAttentionPhaseSession,
)


class _Handler(logging.Handler):
    """Retain emitted phase records without formatting side effects."""

    def __init__(self) -> None:
        """Initialize an empty record list."""

        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append one emitted phase record."""

        self.records.append(record)


def test_phase_session_emits_exact_json_safe_stage_trajectory() -> None:
    """Expose composition, specialization, and consolidation in call order."""

    logger = logging.Logger("tests.standard_unet_attention_phase", logging.DEBUG)
    handler = _Handler()
    logger.addHandler(handler)
    session = StandardUnetAttentionPhaseSession(
        diagnostics=StandardUnetAttentionPhaseDiagnosticsEmitter(logger)
    )

    for sigma in (1.0, 0.5, 0.1):
        with session.activate(_options(sigma)):
            pass

    values = [
        record.__dict__["attention_phase_diagnostics"] for record in handler.records
    ]
    assert [value["stage"] for value in values] == [
        "composition",
        "specialization",
        "consolidation",
    ]
    assert all("restrict_self_attention" not in value for value in values)
    assert all(0.0 <= value["stage_progress"] <= 1.0 for value in values)
    assert all(0.0 <= value["denoising_progress"] <= 1.0 for value in values)
    assert all(
        record.__dict__["operation"] == "unet_attention_coupling.phase"
        for record in handler.records
    )
    json.dumps(values)


def _options(sigma: float) -> dict[str, object]:
    """Return one complete normalized phase call."""

    return {
        "sample_sigmas": torch.linspace(1.0, 0.0, 11),
        "sigmas": torch.tensor([sigma]),
    }
