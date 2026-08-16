# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build anonymous external-inventory fixtures for SDXL visual tests."""

from __future__ import annotations

from pathlib import Path

from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
    VisualAdapterInventory,
    VisualCharacterInventory,
)


def visual_inventory(tmp_path: Path) -> SdxlVisualInventory:
    """Return one complete role-based inventory backed by temporary files."""

    checkpoint = _artifact(tmp_path, "checkpoint.safetensors")
    left = _artifact(tmp_path, "left-character.safetensors")
    right = _artifact(tmp_path, "right-character.safetensors")
    style = _artifact(tmp_path, "style.safetensors")
    return SdxlVisualInventory(
        checkpoint,
        "Checkpoint fixture",
        VisualCharacterInventory(
            left,
            "Left character fixture",
            "left character global prompt",
            "1girl, left character local prompt",
        ),
        VisualCharacterInventory(
            right,
            "Right character fixture",
            "right character global prompt",
            "1girl, right character local prompt",
        ),
        VisualAdapterInventory(
            style,
            "Style fixture",
            "style global trigger",
            "style local trigger",
        ),
    )


def _artifact(root: Path, name: str) -> Path:
    """Write one anonymous placeholder artifact for path validation."""

    path = root / name
    path.write_bytes(b"fixture")
    return path
