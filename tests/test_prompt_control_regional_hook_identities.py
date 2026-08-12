"""Verify stable identities extracted from Prompt Control lazy hook graphs."""

from __future__ import annotations

import pytest

from simple_syrup.runtime.prompt_control_regional_hook_identities import (
    prompt_control_regional_hook_identities,
)


def test_extracts_ordered_create_hook_lora_identity_fields() -> None:
    """Use explicit lazy-node inputs without parsing schedule text."""

    expansion = {
        "text": {"class_type": "PCTextEncode", "inputs": {}},
        "first": {
            "class_type": "CreateHookLora",
            "inputs": {
                "lora_name": "Anima\\style\\adapter-a.safetensors",
                "strength_model": 0.4,
                "strength_clip": 0.25,
            },
        },
        "second": {
            "class_type": "CreateHookLora",
            "inputs": {
                "lora_name": "Anima\\style\\adapter-b.safetensors",
                "strength_model": 1,
                "strength_clip": 0.0,
            },
        },
    }

    assert prompt_control_regional_hook_identities(expansion) == (
        "pc-Anima\\style\\adapter-a.safetensors-0.4-0.25",
        "pc-Anima\\style\\adapter-b.safetensors-1.0-0.0",
    )


@pytest.mark.parametrize(
    ("expansion", "message"),
    [
        ({}, "created no"),
        (
            {
                "hook": {
                    "class_type": "CreateHookLora",
                    "inputs": {
                        "lora_name": "adapter.safetensors",
                        "strength_model": float("nan"),
                        "strength_clip": 1.0,
                    },
                }
            },
            "must be finite",
        ),
        (
            {
                str(index): {
                    "class_type": "CreateHookLora",
                    "inputs": {
                        "lora_name": "adapter.safetensors",
                        "strength_model": 1.0,
                        "strength_clip": 1.0,
                    },
                }
                for index in range(2)
            },
            "must be unique",
        ),
    ],
)
def test_rejects_missing_malformed_or_ambiguous_identity_evidence(
    expansion: object,
    message: str,
) -> None:
    """Fail closed when the lazy graph cannot identify every adapter."""

    with pytest.raises((TypeError, ValueError), match=message):
        prompt_control_regional_hook_identities(expansion)
