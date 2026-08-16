# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify focused multiple-adapter left-region visual declarations."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase
from tools.sdxl_attention_coupling_integration.visual_cases import visual_cases


def test_multiple_left_acceptance_pair_uses_parity_prompt_weight(
    tmp_path: Path,
) -> None:
    """Keep the accepted pair on the proven Attention Couple prompt blend."""

    by_id = {case.case_id: case for case in visual_cases(visual_inventory(tmp_path))}
    prompt_control = by_id["multiple-left-style-prompt-control"]
    control = by_id["multiple-left-style-only-control"]
    candidate = by_id["multiple-left"]

    assert (
        prompt_control.regional_prompt_weight
        == control.regional_prompt_weight
        == candidate.regional_prompt_weight
        == 0.4
    )
    assert prompt_control.left_l == control.left_l
    assert prompt_control.left_g == control.left_g
    assert prompt_control.right_l == control.right_l
    assert prompt_control.right_g == control.right_g
    assert prompt_control.global_adapters == control.global_adapters == ()
    assert prompt_control.left_adapters == ()
    assert len(control.left_adapters) == 1
    assert control.left_l == candidate.left_l
    assert control.left_g == candidate.left_g
    assert control.right_l == candidate.right_l
    assert control.right_g == candidate.right_g
    assert control.global_adapters == candidate.global_adapters == ()
    assert control.left_adapters == candidate.left_adapters[1:]
    assert len(candidate.left_adapters) == 2
    assert candidate.left_adapters[0].lora_name == LEFT_CHARACTER_SELECTION
    assert control.left_adapters[0].lora_name == STYLE_SELECTION
    assert control.left_adapters[0].model_strength == 0.55
    assert control.left_adapters[0].clip_strength == 0.55
    assert control.right_adapters == candidate.right_adapters == ()
    assert prompt_control.mask_profile is control.mask_profile is candidate.mask_profile
    assert prompt_control.region_mask_feather == control.region_mask_feather == 0
    assert prompt_control.modes == control.modes == candidate.modes


def test_model_only_style_ablation_changes_only_clip_strength(tmp_path: Path) -> None:
    """Preserve full model-side style work while isolating CLIP causality."""

    by_id = {case.case_id: case for case in visual_cases(visual_inventory(tmp_path))}
    full = by_id["multiple-left-style-only-control"]
    model_only = by_id["multiple-left-model-only-style"]

    _assert_same_multiple_left_inputs(full, model_only)
    full_adapter = full.left_adapters[0]
    model_only_adapter = model_only.left_adapters[0]
    assert full_adapter.lora_name == model_only_adapter.lora_name == STYLE_SELECTION
    assert full_adapter.model_strength == model_only_adapter.model_strength == 0.55
    assert full_adapter.clip_strength == 0.55
    assert model_only_adapter.clip_strength == 0.0
    assert full_adapter.schedule == model_only_adapter.schedule == ((0.0, 1.0),)
    assert full.regional_prompt_weight == 0.4
    assert model_only.regional_prompt_weight == 1.0


def test_scheduled_style_changes_only_schedule_and_diagnostic_weight(
    tmp_path: Path,
) -> None:
    """Retain the historical full-strength schedule diagnostic explicitly."""

    by_id = {case.case_id: case for case in visual_cases(visual_inventory(tmp_path))}
    full = by_id["multiple-left-style-only-control"]
    scheduled = by_id["multiple-left-scheduled-style-only"]

    _assert_same_multiple_left_inputs(full, scheduled)
    assert full.left_adapters[0].lora_name == scheduled.left_adapters[0].lora_name
    assert full.left_adapters[0].model_strength == 0.55
    assert scheduled.left_adapters[0].model_strength == 0.55
    assert full.left_adapters[0].clip_strength == 0.55
    assert scheduled.left_adapters[0].clip_strength == 0.55
    assert full.left_adapters[0].schedule == ((0.0, 1.0),)
    assert scheduled.left_adapters[0].schedule == (
        (0.0, 0.0),
        (0.3, 0.35),
        (0.45, 1.0),
    )
    assert full.regional_prompt_weight == 0.4
    assert scheduled.regional_prompt_weight == 1.0


def test_midpoint_style_changes_only_schedule_timing(tmp_path: Path) -> None:
    """Increase style exposure while preserving terminal strength and inputs."""

    by_id = {case.case_id: case for case in visual_cases(visual_inventory(tmp_path))}
    safe = by_id["multiple-left-scheduled-style-only"]
    midpoint = by_id["multiple-left-midpoint-style-only"]

    _assert_same_multiple_left_inputs(safe, midpoint)
    assert safe.left_adapters[0].lora_name == midpoint.left_adapters[0].lora_name
    assert safe.left_adapters[0].model_strength == 0.55
    assert midpoint.left_adapters[0].model_strength == 0.55
    assert safe.left_adapters[0].clip_strength == 0.55
    assert midpoint.left_adapters[0].clip_strength == 0.55
    assert safe.left_adapters[0].schedule == (
        (0.0, 0.0),
        (0.3, 0.35),
        (0.45, 1.0),
    )
    assert midpoint.left_adapters[0].schedule == (
        (0.0, 0.0),
        (0.15, 0.35),
        (0.3, 1.0),
    )
    assert safe.regional_prompt_weight == midpoint.regional_prompt_weight == 1.0


def test_minimal_window_style_changes_only_schedule_timing(tmp_path: Path) -> None:
    """Expand style exposure without changing identity or terminal strength."""

    by_id = {case.case_id: case for case in visual_cases(visual_inventory(tmp_path))}
    midpoint = by_id["multiple-left-midpoint-style-only"]
    minimal = by_id["multiple-left-minimal-window-style-only"]

    _assert_same_multiple_left_inputs(midpoint, minimal)
    assert midpoint.left_adapters[0].lora_name == minimal.left_adapters[0].lora_name
    assert midpoint.left_adapters[0].model_strength == 0.55
    assert minimal.left_adapters[0].model_strength == 0.55
    assert midpoint.left_adapters[0].clip_strength == 0.55
    assert minimal.left_adapters[0].clip_strength == 0.55
    assert midpoint.left_adapters[0].schedule == (
        (0.0, 0.0),
        (0.15, 0.35),
        (0.3, 1.0),
    )
    assert minimal.left_adapters[0].schedule == (
        (0.0, 0.0),
        (0.05, 0.35),
        (0.15, 1.0),
    )
    assert midpoint.regional_prompt_weight == minimal.regional_prompt_weight == 1.0


def _assert_same_multiple_left_inputs(
    first: SdxlVisualCase,
    second: SdxlVisualCase,
) -> None:
    """Assert shared visual-case inputs through their typed public attributes."""

    assert first.left_l == second.left_l
    assert first.left_g == second.left_g
    assert first.right_l == second.right_l
    assert first.right_g == second.right_g
    assert first.base_positive_g == second.base_positive_g
    assert first.base_positive_l == second.base_positive_l
    assert first.global_style_l == second.global_style_l
    assert first.global_style_g == second.global_style_g
    assert first.global_adapters == second.global_adapters == ()
    assert len(first.left_adapters) == len(second.left_adapters) == 1
    assert first.right_adapters == second.right_adapters == ()
    assert first.mask_profile is second.mask_profile
    assert first.region_mask_feather == second.region_mask_feather
    assert first.modes == second.modes
