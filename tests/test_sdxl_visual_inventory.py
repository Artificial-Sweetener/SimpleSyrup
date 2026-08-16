# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify external SDXL visual artifact inventory validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)


def test_inventory_loads_complete_anonymous_roles(tmp_path: Path) -> None:
    """Load external artifact paths, labels, and triggers without fixed identities."""

    expected = visual_inventory(tmp_path)
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint_source": str(expected.checkpoint_source),
                "checkpoint_label": expected.checkpoint_label,
                "left_character": {
                    "source": str(expected.left_character.source),
                    "label": expected.left_character.label,
                    "prompt_g": expected.left_character.prompt_g,
                    "prompt_l": expected.left_character.prompt_l,
                },
                "right_character": {
                    "source": str(expected.right_character.source),
                    "label": expected.right_character.label,
                    "prompt_g": expected.right_character.prompt_g,
                    "prompt_l": expected.right_character.prompt_l,
                },
                "style": {
                    "source": str(expected.style.source),
                    "label": expected.style.label,
                    "prompt_g": expected.style.prompt_g,
                    "prompt_l": expected.style.prompt_l,
                },
            }
        ),
        encoding="utf-8",
    )

    observed = SdxlVisualInventory.load(path)

    assert observed == expected


def test_inventory_rejects_missing_external_artifacts(tmp_path: Path) -> None:
    """Fail before managed execution when an externally selected file is absent."""

    inventory = visual_inventory(tmp_path)
    inventory.style.source.unlink()

    with pytest.raises(FileNotFoundError, match="adapter source does not exist"):
        SdxlVisualInventory(
            inventory.checkpoint_source,
            inventory.checkpoint_label,
            inventory.left_character,
            inventory.right_character,
            inventory.style,
        )
