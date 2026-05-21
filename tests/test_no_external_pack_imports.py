# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests that SimpleSyrup does not couple to optional node packs."""

from __future__ import annotations

from pathlib import Path


def test_simple_syrup_does_not_import_impact_or_layerstyle() -> None:
    """Runtime source should not import Impact Pack or LayerStyle modules."""

    project_root = Path(__file__).resolve().parents[1]
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (project_root / "simple_syrup").rglob("*.py")
    )

    forbidden = (
        "ComfyUI_LayerStyle_Advance",
        "comfyui_layerstyle",
        "impact.core",
        "local_groundingdino",
        "from impact",
        "import impact",
    )
    for token in forbidden:
        assert token not in source_text
