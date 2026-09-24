# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build shared Prompt Control characterization evidence for tests."""

from __future__ import annotations

import uuid

from tools.prompt_control_characterization.cases import PromptControlCase
from tools.prompt_control_characterization.history_outputs import PromptControlOutputs


def prompt_control_outputs(case: PromptControlCase) -> PromptControlOutputs:
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
