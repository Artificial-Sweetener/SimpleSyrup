# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact benchmark-only interop validation profiling."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import torch

from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_model_capabilities import RegionalModelCapabilities
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropReport,
    RegionalModelPatchInteropValidator,
)
from tools.attention_coupling_benchmark.comfy_probe.interop_validation_profile import (
    ProfiledRegionalModelPatchInteropValidator,
)

_LOGGER_NAME = "simple_syrup.runtime.regional_lora.standard_unet_cold_path"


def test_profiled_interop_validator_preserves_both_exact_delegates(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """Retain arguments, report identity, result, and ordered timing stages."""

    report = cast(RegionalModelPatchInteropReport, object())
    capabilities = cast(RegionalModelCapabilities, object())
    processed = cast(ProcessedRegionalAttentionPlan, object())
    model = SimpleNamespace(load_device=torch.device("cpu"))
    calls: list[tuple[str, tuple[object, ...]]] = []

    def validate(
        _owner: object,
        supplied_model: object,
        supplied_capabilities: RegionalModelCapabilities,
    ) -> RegionalModelPatchInteropReport:
        calls.append(("validate", (supplied_model, supplied_capabilities)))
        return report

    def validate_execution(
        _owner: object,
        supplied_report: RegionalModelPatchInteropReport,
        supplied_plan: ProcessedRegionalAttentionPlan,
        supplied_mode: RegionalAttentionExecutionMode,
    ) -> None:
        calls.append(
            ("validate_execution", (supplied_report, supplied_plan, supplied_mode))
        )

    monkeypatch.setattr(RegionalModelPatchInteropValidator, "validate", validate)
    monkeypatch.setattr(
        RegionalModelPatchInteropValidator,
        "validate_execution",
        validate_execution,
    )
    validator = ProfiledRegionalModelPatchInteropValidator()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        actual = validator.validate(model, capabilities)
        validator.validate_execution(
            report,
            processed,
            RegionalAttentionExecutionMode.FULL,
        )

    assert actual is report
    assert calls == [
        ("validate", (model, capabilities)),
        (
            "validate_execution",
            (report, processed, RegionalAttentionExecutionMode.FULL),
        ),
    ]
    assert [
        record.cold_path_diagnostics["stage"]
        for record in caplog.records
        if hasattr(record, "cold_path_diagnostics")
    ] == ["interop_validation", "interop_execution_validation"]
