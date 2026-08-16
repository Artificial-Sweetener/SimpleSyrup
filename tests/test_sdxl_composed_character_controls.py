# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify generic one-adapter isolation for composed character evidence."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.visual_cases import visual_cases


def test_left_adapter_control_preserves_exact_simultaneous_prompt_topology(
    tmp_path: Path,
) -> None:
    """Change only right-adapter absence against the simultaneous visual case."""

    by_id = {case.case_id: case for case in visual_cases(visual_inventory(tmp_path))}
    control = by_id["different-characters-left-adapter-control"]
    simultaneous = by_id["different-characters"]

    assert control.left_l == simultaneous.left_l
    assert control.left_g == simultaneous.left_g
    assert control.right_l == simultaneous.right_l
    assert control.right_g == simultaneous.right_g
    assert control.base_positive_g == simultaneous.base_positive_g
    assert control.base_positive_l == simultaneous.base_positive_l
    assert control.regional_prompt_weight == simultaneous.regional_prompt_weight == 0.4
    assert control.mask_profile is simultaneous.mask_profile
    assert control.region_mask_feather == simultaneous.region_mask_feather == 0
    assert control.left_adapters == simultaneous.left_adapters
    assert len(control.left_adapters) == 1
    assert control.right_adapters == ()
    assert len(simultaneous.right_adapters) == 1
