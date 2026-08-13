# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the fixed native-SDXL visual acceptance declarations."""

from __future__ import annotations

from tools.sdxl_attention_coupling_integration.visual_cases import (
    character_b_NAME,
    ELDEN_STYLE_NAME,
    VisualMaskProfile,
    VisualMode,
    visual_cases,
)


def test_visual_matrix_has_twelve_cases_and_fourteen_outputs() -> None:
    """Cover every required scenario without multiplying upscale refinements."""

    cases = visual_cases()

    assert len(cases) == 12
    assert sum(len(case.modes) for case in cases) == 14
    assert len({case.case_id for case in cases}) == len(cases)
    assert cases[0].case_id == "baseline"
    assert cases[0].global_adapters == ()
    assert cases[0].left_adapters == ()
    assert cases[0].right_adapters == ()


def test_global_regional_and_schedule_contracts_remain_distinct() -> None:
    """Preserve ordinary global use, authored regional reuse, and keyframes."""

    by_id = {case.case_id: case for case in visual_cases()}
    combined = by_id["global-style-regional-character"]
    assert [adapter.lora_name for adapter in combined.global_adapters] == [
        ELDEN_STYLE_NAME
    ]
    assert [adapter.lora_name for adapter in combined.right_adapters] == [character_b_NAME]
    assert combined.modes == (
        VisualMode.FULL,
        VisualMode.TILED,
        VisualMode.CONTEXTUAL,
    )
    same = by_id["same-style-both"]
    assert same.left_adapters[0].lora_name == same.right_adapters[0].lora_name
    scheduled = by_id["scheduled-right-character"].right_adapters[0]
    assert scheduled.schedule == ((0.0, 1.0), (0.6, 0.0))
    assert by_id["soft-overlap"].mask_profile is VisualMaskProfile.SOFT_OVERLAP
    assert by_id["uncovered-center"].mask_profile is VisualMaskProfile.UNCOVERED_CENTER
