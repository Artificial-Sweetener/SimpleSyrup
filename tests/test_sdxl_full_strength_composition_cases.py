# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lock the bounded full-strength two-region character-LoRA matrix."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_full_strength_lora_fidelity.composition_cases import (
    full_strength_composition_cases,
)


def test_full_strength_composition_cases_are_constant_and_ordered(
    tmp_path: Path,
) -> None:
    """Require left-only, right-only, then simultaneous exact adapters."""

    inventory = visual_inventory(tmp_path)
    prompts = SdxlVisualPromptSet(*tuple(f"prompt-{index}" for index in range(12)))
    cases = full_strength_composition_cases(inventory, prompts)

    assert tuple(case.case_id for case in cases) == (
        "full-strength-left-only",
        "full-strength-right-only",
        "full-strength-simultaneous",
    )
    assert tuple(adapter.lora_name for adapter in cases[0].left_adapters) == (
        LEFT_CHARACTER_SELECTION,
    )
    assert tuple(adapter.lora_name for adapter in cases[1].right_adapters) == (
        RIGHT_CHARACTER_SELECTION,
    )
    assert cases[2].left_adapters == cases[0].left_adapters
    assert cases[2].right_adapters == cases[1].right_adapters
    for case in cases:
        for adapter in (*case.left_adapters, *case.right_adapters):
            assert adapter.model_strength == 1.0
            assert adapter.clip_strength == 1.0
            assert adapter.schedule == ((0.0, 1.0),)
        assert case.regional_prompt_weight == 1.0
