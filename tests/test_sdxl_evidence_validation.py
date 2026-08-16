# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify SDXL evidence validation for persistent regional variants."""

from __future__ import annotations

import pytest

from tools.sdxl_attention_coupling_integration.evidence_validation import (
    validate_sdxl_diagnostics,
)


def test_full_mode_accepts_persistent_variant_composition_diagnostics() -> None:
    """Accept exact phase/schedule evidence when packed snapshots do not apply."""

    diagnostics = {
        "record_count": 0,
        "snapshots": [],
        "composition_record_count": 2,
        "composition": [
            {
                "stage": "composition",
                "denoising_progress": 0.0,
                "sampling_sigma": 14.0,
                "schedule_multipliers": [1.0, 1.0],
                "adapter_schedules": [_schedule(1.0), _schedule(1.0)],
            },
            {
                "stage": "specialization",
                "denoising_progress": 0.5,
                "sampling_sigma": 3.0,
                "schedule_multipliers": [1.0, 1.0],
                "adapter_schedules": [_schedule(1.0), _schedule(1.0)],
            },
        ],
    }

    assert (
        validate_sdxl_diagnostics(
            diagnostics,
            label="persistent",
            expected_spatial_modes=frozenset({"full"}),
        )
        is diagnostics
    )


def test_non_full_mode_still_requires_spatial_snapshots() -> None:
    """Do not infer tiled or Contextual coverage from full-mode composition."""

    with pytest.raises(ValueError, match="spatial-mode snapshots"):
        validate_sdxl_diagnostics(
            {
                "record_count": 0,
                "snapshots": [],
                "composition_record_count": 1,
                "composition": [
                    {
                        "stage": "composition",
                        "denoising_progress": 0.0,
                        "sampling_sigma": 14.0,
                        "schedule_multipliers": [1.0],
                        "adapter_schedules": [_schedule(1.0)],
                    }
                ],
            },
            label="tiled",
            expected_spatial_modes=frozenset({"tile"}),
        )


def test_full_mode_rejects_missing_sigma_domain_schedule_evidence() -> None:
    """Do not accept linear progress as proof of a converted Comfy boundary."""

    with pytest.raises(ValueError, match="composition sigma"):
        validate_sdxl_diagnostics(
            {
                "record_count": 0,
                "snapshots": [],
                "composition_record_count": 1,
                "composition": [
                    {
                        "stage": "composition",
                        "denoising_progress": 0.0,
                        "schedule_multipliers": [1.0],
                    }
                ],
            },
            label="persistent",
            expected_spatial_modes=frozenset({"full"}),
        )


def _schedule(strength: float) -> list[dict[str, float | int]]:
    """Return one anonymous converted schedule fixture."""

    return [
        {
            "start_percent": 0.0,
            "start_sigma": 14.0,
            "strength_multiplier": strength,
            "guarantee_steps": 0,
        }
    ]
