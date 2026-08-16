# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify native-self-attention standard-UNet diagnostic requirements."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tools.sdxl_attention_coupling_integration.visual_diagnostic_validation import (
    validate_standard_unet_diagnostics,
)


def test_validation_requires_all_phases_and_native_global_self_attention() -> None:
    """Accept a complete trajectory with no standard attn1 replacement."""

    observed = _diagnostics()

    assert validate_standard_unet_diagnostics(observed) is observed


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("attention_phase_record_count", 0, "attention-phase records"),
        ("attention_phases", [], "attention-phase values"),
    ),
)
def test_validation_rejects_missing_runtime_proof(
    field: str,
    value: object,
    match: str,
) -> None:
    """Fail closed when capability discovery lacks execution evidence."""

    diagnostics = deepcopy(_diagnostics())
    diagnostics[field] = value

    with pytest.raises(ValueError, match=match):
        validate_standard_unet_diagnostics(diagnostics)


def test_validation_rejects_any_standard_self_attention_replacement() -> None:
    """Prove standard UNet retains native global self-attention."""

    counted = deepcopy(_diagnostics())
    counted["self_attention_record_count"] = 1
    with pytest.raises(ValueError, match="zero self-attention records"):
        validate_standard_unet_diagnostics(counted)

    populated = deepcopy(_diagnostics())
    populated["self_attention"] = [{"stage": "specialization"}]
    with pytest.raises(ValueError, match="no self-attention values"):
        validate_standard_unet_diagnostics(populated)


def test_validation_rejects_incomplete_stage_evidence() -> None:
    """Require the complete phase trajectory around native self-attention."""

    incomplete = deepcopy(_diagnostics())
    incomplete["attention_phases"] = [
        {"stage": "composition"},
        {"stage": "specialization"},
    ]
    with pytest.raises(ValueError, match="complete phase trajectory"):
        validate_standard_unet_diagnostics(incomplete)


def _diagnostics() -> dict[str, object]:
    """Return generic complete packed execution evidence."""

    return {
        "attention_phase_record_count": 30,
        "attention_phases": [
            {"stage": "composition"},
            {"stage": "specialization"},
            {"stage": "consolidation"},
        ],
        "self_attention_record_count": 0,
        "self_attention": [],
    }
