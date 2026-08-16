# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the full-strength character-LoRA reference matrix."""

from __future__ import annotations

from tools.sdxl_full_strength_lora_fidelity.cases import (
    CharacterFidelitySpec,
    FidelityExecutionMode,
    fidelity_cases,
)


def test_matrix_pairs_global_and_all_one_regional_execution_per_character() -> None:
    """Compare identical character controls at constant authored strength one."""

    cases = fidelity_cases(
        (
            CharacterFidelitySpec("first", "First", "first.safetensors", "one", "one"),
            CharacterFidelitySpec(
                "second",
                "Second",
                "second.safetensors",
                "two",
                "two",
            ),
        )
    )

    assert [case.mode for case in cases] == [
        FidelityExecutionMode.GLOBAL_REFERENCE,
        FidelityExecutionMode.REGIONAL_ALL_ONE,
        FidelityExecutionMode.GLOBAL_REFERENCE,
        FidelityExecutionMode.REGIONAL_ALL_ONE,
    ]
    assert all(case.model_strength == 1.0 for case in cases)
    assert all(case.clip_strength == 1.0 for case in cases)
    assert all(case.schedule == ((0.0, 1.0),) for case in cases)
