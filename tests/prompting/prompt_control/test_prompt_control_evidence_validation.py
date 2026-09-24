# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact Prompt Control characterization acceptance policy."""

from __future__ import annotations

import pytest

from tools.prompt_control_characterization.cases import PromptControlCase, cases
from tools.prompt_control_characterization.evidence_validation import (
    validate_evidence,
)

from .support.evidence import prompt_control_outputs


def test_static_text_evidence_accepts_exact_entries_and_stable_uuids() -> None:
    """Accept the base schedule while proving UUIDv4 entry identity."""

    case = _case("text-static")
    outputs = prompt_control_outputs(case)
    validate_evidence(case, outputs)


def test_evidence_rejects_conditioning_boundary_drift() -> None:
    """Fail when Prompt Control changes an exact range endpoint."""

    case = _case("text-static")
    outputs = prompt_control_outputs(case)
    positive = outputs.snapshot["positive"]
    assert isinstance(positive, list)
    first = positive[0]
    assert isinstance(first, dict)
    metadata = first["metadata"]
    assert isinstance(metadata, dict)
    metadata["end_percent"] = 0.9
    with pytest.raises(ValueError, match="end_percent"):
        validate_evidence(case, outputs)


def _case(case_id: str) -> PromptControlCase:
    """Return one case by stable identity."""

    return next(case for case in cases() if case.case_id == case_id)
