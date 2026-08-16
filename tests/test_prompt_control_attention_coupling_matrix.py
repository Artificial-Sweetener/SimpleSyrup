# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the immutable P9.1 matrix and validated P0.8 dependency."""

from tools.prompt_control_attention_coupling_integration.baseline import load_baseline
from tools.prompt_control_attention_coupling_integration.matrix import (
    CFG,
    HEIGHT,
    STEPS,
    WIDTH,
    cases,
)
from tools.prompt_control_characterization.cases import cases as p0_8_cases


def test_matrix_wraps_all_p0_8_cases_without_redefining_schedule_values() -> None:
    """Preserve exact P0.8 case object identities and fixed sampler controls."""

    definitions = cases()

    assert tuple(case.characterization for case in definitions) == p0_8_cases()
    assert tuple(case.case_id for case in definitions) == tuple(
        case.case_id for case in p0_8_cases()
    )
    assert len(definitions) == 9
    assert {case.cfg for case in definitions} == {CFG}
    assert {case.feather for case in definitions} == {0}
    assert (WIDTH, HEIGHT, STEPS) == (512, 512, 8)
    assert definitions[4].adapter_identities == ("primary_adapter-static-0.75",)
    assert definitions[-1].adapter_identities == ()


def test_baseline_revalidates_all_nine_pinned_observations() -> None:
    """Require the exact external P0.8 artifact before P9.1 acceptance."""

    baseline = load_baseline()

    assert len(baseline.observations) == 9
    assert len(baseline.sha256) == 64
    assert tuple(item.case.case_id for item in baseline.observations) == tuple(
        case.case_id for case in cases()
    )
