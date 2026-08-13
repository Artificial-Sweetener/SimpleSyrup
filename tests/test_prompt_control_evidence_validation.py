# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact Prompt Control characterization acceptance policy."""

from __future__ import annotations

import uuid

import pytest

from tools.prompt_control_characterization.cases import PromptControlCase, cases
from tools.prompt_control_characterization.evidence_validation import (
    validate_evidence,
)
from tools.prompt_control_characterization.history_outputs import PromptControlOutputs


def test_static_text_evidence_accepts_exact_entries_and_stable_uuids() -> None:
    """Accept the base schedule while proving UUIDv4 entry identity."""

    case = _case("text-static")
    outputs = _outputs(case)
    validate_evidence(case, outputs)


def test_evidence_rejects_conditioning_boundary_drift() -> None:
    """Fail when Prompt Control changes an exact range endpoint."""

    case = _case("text-static")
    outputs = _outputs(case)
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


def _outputs(case: PromptControlCase) -> PromptControlOutputs:
    """Build minimal complete text-only host evidence."""

    negative_uuids = [str(uuid.uuid4()) for _ in case.expected_negative]
    positive_uuids = [str(uuid.uuid4()) for _ in case.expected_positive]
    positive = [
        {
            "index": index,
            "tensor": {},
            "metadata": {
                "start_percent": expected.start,
                "end_percent": expected.end,
                **(
                    {"strength": expected.strength}
                    if expected.strength is not None
                    else {}
                ),
            },
        }
        for index, expected in enumerate(case.expected_positive)
    ]
    calls = [
        {
            "call_index": index,
            "timestep": 1.0 - index / 10,
            "uuids": [*negative_uuids, *positive_uuids],
            "cond_or_uncond": [
                *([1] * len(negative_uuids)),
                *([0] * len(positive_uuids)),
            ],
            "effective_strengths": [],
        }
        for index in range(1, case.expected_model_call_count + 1)
    ]
    return PromptControlOutputs(
        expansion={
            "case_id": case.case_id,
            "text_expansion": {
                "node-1": {"class_type": "PCTextEncode", "inputs": {}},
                "node-2": {
                    "class_type": "ConditioningSetTimestepRange",
                    "inputs": {},
                },
            },
            "lora_expansion": {},
        },
        snapshot={
            "case_id": case.case_id,
            "positive": positive,
            "negative": [
                {
                    "index": index,
                    "tensor": {},
                    "metadata": {
                        "start_percent": expected.start,
                        "end_percent": expected.end,
                    },
                }
                for index, expected in enumerate(case.expected_negative)
            ],
            "hooks": [],
        },
        runtime={
            "run_id": case.case_id,
            "model_call_count": case.expected_model_call_count,
            "static_patch_target_count": 0,
            "static_patch_entry_count_histogram": {},
            "registered_hooks": [],
            "schedule_transitions": [
                {
                    **calls[call_index - 1],
                    "effective_strengths": list(strengths),
                }
                for call_index, strengths in case.expected_runtime_transitions
            ],
            "calls": calls,
        },
    )
