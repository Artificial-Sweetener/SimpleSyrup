# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the requested global-style and regional-character proof matrix."""

from __future__ import annotations

from tools.global_style_character_proof.matrix import cases


def test_matrix_keeps_character_regional_while_global_style_strength_changes() -> None:
    """Compare one character control against two full-image style strengths."""

    definitions = cases()

    assert [case.case_id for case in definitions] == [
        "character-right-only",
        "global-style-050-character-right",
        "global-style-100-character-right",
    ]
    assert [case.global_style_strength for case in definitions] == [None, 0.5, 1.0]
