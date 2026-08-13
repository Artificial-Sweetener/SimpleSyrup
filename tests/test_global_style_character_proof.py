# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the requested global-style and regional-character proof matrix."""

from __future__ import annotations

from tools.global_style_character_proof.matrix import cases


def test_matrix_keeps_character_a_regional_while_global_adapter_a_strength_changes() -> None:
    """Compare one character control against two full-image style strengths."""

    definitions = cases()

    assert [case.case_id for case in definitions] == [
        "character_a-right-only",
        "global-adapter_a-050-character_a-right",
        "global-adapter_a-100-character_a-right",
    ]
    assert [case.global_style_strength for case in definitions] == [None, 0.5, 1.0]
