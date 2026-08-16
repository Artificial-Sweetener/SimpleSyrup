# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify benchmark-only structured regional diagnostic capture."""

from __future__ import annotations

import logging

import pytest

from tools.attention_coupling_benchmark.comfy_probe import (
    regional_diagnostics_capture,
)

CaptureRegionalDiagnosticsV3 = regional_diagnostics_capture.CaptureRegionalDiagnosticsV3
ReadRegionalDiagnosticsV3 = regional_diagnostics_capture.ReadRegionalDiagnosticsV3

_ANIMA_LOGGER_NAME = "simple_syrup.runtime.regional_lora.anima_diagnostics"
_ANIMA_ATTENTION_LOGGER_NAME = (
    "simple_syrup.runtime.attention_coupling.anima_diagnostics"
)
_UNET_LOGGER_NAME = "simple_syrup.runtime.attention_coupling.unet_diagnostics"
_UNET_PHASE_LOGGER_NAME = "simple_syrup.runtime.attention_coupling.unet_attention_phase"
_UNET_COMPOSITION_LOGGER_NAME = (
    "simple_syrup.runtime.regional_lora.standard_unet_composition"
)


def test_capture_returns_unique_json_safe_snapshots_and_exact_record_count() -> None:
    """Capture target records, deduplicate adjacent snapshots, and pass through."""

    model = object()
    latent = {"samples": object()}
    logger = logging.getLogger(_ANIMA_LOGGER_NAME)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        started = CaptureRegionalDiagnosticsV3.execute(model, "capture-1")
        payload = {
            "spatial_mode": "tile",
            "canonical_mask": {"regions": [{"nonzero_fraction": 0.5}]},
        }
        logger.info("captured", extra={"regional_diagnostics": payload})
        logger.info("captured again", extra={"regional_diagnostics": payload})
        payload["spatial_mode"] = "contextual_global"
        logger.info("captured global", extra={"regional_diagnostics": payload})
        result = ReadRegionalDiagnosticsV3.execute(latent, "capture-1")
    finally:
        logger.setLevel(previous_level)

    assert started.result[0] is model
    assert result.result[0] is latent
    capture = result.ui["regional_diagnostics"][0]
    assert capture["record_count"] == 3
    assert [item["spatial_mode"] for item in capture["snapshots"]] == [
        "tile",
        "contextual_global",
    ]


def test_capture_rejects_duplicate_start_and_missing_read() -> None:
    """Keep each capture identity single-owner and fail closed on missing state."""

    CaptureRegionalDiagnosticsV3.execute(object(), "capture-duplicate")
    with pytest.raises(ValueError, match="already active"):
        CaptureRegionalDiagnosticsV3.execute(object(), "capture-duplicate")
    with pytest.raises(ValueError, match="did not observe"):
        ReadRegionalDiagnosticsV3.execute({}, "capture-duplicate")
    with pytest.raises(ValueError, match="was not started"):
        ReadRegionalDiagnosticsV3.execute({}, "capture-duplicate")


def test_capture_schemas_remain_benchmark_only_v3_nodes() -> None:
    """Expose focused dev-only start and terminal read contracts."""

    capture = CaptureRegionalDiagnosticsV3.define_schema()
    read = ReadRegionalDiagnosticsV3.define_schema()

    assert capture.node_id == "SimpleSyrupBenchmark.CaptureRegionalDiagnostics"
    assert capture.is_dev_only is True
    assert read.node_id == "SimpleSyrupBenchmark.ReadRegionalDiagnostics"
    assert read.is_dev_only is True
    assert read.is_output_node is True


def test_capture_accepts_standard_unet_regional_diagnostics() -> None:
    """Capture the standard-UNet emitter through the same JSON-safe boundary."""

    logger = logging.getLogger(_UNET_LOGGER_NAME)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        CaptureRegionalDiagnosticsV3.execute(object(), "capture-unet")
        logger.info(
            "captured UNet resolution",
            extra={
                "regional_diagnostics": {
                    "strategy": "attention_coupling",
                    "backend": "comfy.ldm.diffusionmodules.openaimodel.UNetModel",
                    "denoiser_call_multiplier": 1.0,
                }
            },
        )
        result = ReadRegionalDiagnosticsV3.execute({}, "capture-unet")
    finally:
        logger.setLevel(previous_level)

    capture = result.ui["regional_diagnostics"][0]
    assert capture["record_count"] == 1
    assert capture["snapshots"][0]["denoiser_call_multiplier"] == 1.0


def test_capture_leases_debug_level_and_restores_the_exact_previous_level() -> None:
    """Capture detailed records without enabling normal runtime DEBUG logging."""

    logger = logging.getLogger(_UNET_LOGGER_NAME)
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        CaptureRegionalDiagnosticsV3.execute(object(), "capture-debug-lease")
        assert logger.level == logging.DEBUG
        logger.debug(
            "captured detailed UNet resolution",
            extra={"regional_diagnostics": {"stage": "debug-evidence"}},
        )
        result = ReadRegionalDiagnosticsV3.execute({}, "capture-debug-lease")
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(previous_level)

    capture = result.ui["regional_diagnostics"][0]
    assert capture["record_count"] == 1
    assert capture["snapshots"] == [{"stage": "debug-evidence"}]


def test_capture_keeps_standard_unet_attention_phases_separate() -> None:
    """Retain call phases without changing regional resolution snapshots."""

    logger = logging.getLogger(_UNET_PHASE_LOGGER_NAME)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        CaptureRegionalDiagnosticsV3.execute(object(), "capture-unet-phase")
        logger.info(
            "captured UNet phase",
            extra={
                "attention_phase_diagnostics": {
                    "stage": "specialization",
                    "denoising_progress": 0.25,
                    "stage_progress": 0.27,
                    "restrict_self_attention": True,
                }
            },
        )
        result = ReadRegionalDiagnosticsV3.execute({}, "capture-unet-phase")
    finally:
        logger.setLevel(previous_level)

    capture = result.ui["regional_diagnostics"][0]
    assert capture["record_count"] == 0
    assert capture["snapshots"] == []
    assert capture["attention_phase_record_count"] == 1
    assert capture["attention_phases"] == [
        {
            "stage": "specialization",
            "denoising_progress": 0.25,
            "stage_progress": 0.27,
            "restrict_self_attention": True,
        }
    ]


def test_capture_accepts_attention_only_anima_diagnostics() -> None:
    """Capture common Anima evidence when no regional model LoRA is present."""

    logger = logging.getLogger(_ANIMA_ATTENTION_LOGGER_NAME)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        CaptureRegionalDiagnosticsV3.execute(object(), "capture-anima-attention")
        logger.info(
            "captured attention-only Anima execution",
            extra={
                "regional_diagnostics": {
                    "strategy": "attention_coupling",
                    "model_call": {"sampling_sigma": 0.5},
                }
            },
        )
        result = ReadRegionalDiagnosticsV3.execute({}, "capture-anima-attention")
    finally:
        logger.setLevel(previous_level)

    capture = result.ui["regional_diagnostics"][0]
    assert capture["record_count"] == 1
    assert capture["snapshots"][0]["model_call"]["sampling_sigma"] == 0.5


def test_capture_keeps_persistent_variant_composition_separate() -> None:
    """Capture model-active standard-UNet phase and schedule evidence."""

    logger = logging.getLogger(_UNET_COMPOSITION_LOGGER_NAME)
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        CaptureRegionalDiagnosticsV3.execute(object(), "capture-composition")
        logger.debug(
            "captured composition",
            extra={
                "regional_composition": {
                    "stage": "specialization",
                    "denoising_progress": 0.5,
                    "schedule_multipliers": [1.0, 1.0],
                }
            },
        )
        result = ReadRegionalDiagnosticsV3.execute({}, "capture-composition")
    finally:
        logger.setLevel(previous_level)

    capture = result.ui["regional_diagnostics"][0]
    assert capture["record_count"] == 0
    assert capture["composition_record_count"] == 1
    assert capture["composition"] == [
        {
            "stage": "specialization",
            "denoising_progress": 0.5,
            "schedule_multipliers": [1.0, 1.0],
        }
    ]
