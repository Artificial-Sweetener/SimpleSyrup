# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify deterministic temporary Phase 8 SDXL mask ownership."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.sdxl_attention_coupling_integration.masks import ManagedSdxlSplitMasks
from tools.sdxl_attention_coupling_integration.matrix import (
    REGION_WIDTH,
    RIGHT_REGION_START,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)


def test_split_masks_leave_exact_uncovered_center_band(tmp_path: Path) -> None:
    """Write separated regional masks and remove only their exact owned files."""

    manager = ManagedSdxlSplitMasks(input_root=tmp_path, run_id="run-1")

    with manager:
        paths = tuple(tmp_path / name for name in manager.names)
        evidence = manager.evidence()
        with Image.open(paths[0]) as left, Image.open(paths[1]) as right:
            assert left.size == right.size == (TARGET_WIDTH, TARGET_HEIGHT)
            assert left.getpixel((REGION_WIDTH - 1, 0)) == (255, 255, 255)
            assert left.getpixel((REGION_WIDTH, 0)) == (0, 0, 0)
            assert right.getpixel((RIGHT_REGION_START - 1, 0)) == (0, 0, 0)
            assert right.getpixel((RIGHT_REGION_START, 0)) == (255, 255, 255)
            center = TARGET_WIDTH // 2
            assert left.getpixel((center, 0)) == (0, 0, 0)
            assert right.getpixel((center, 0)) == (0, 0, 0)
        assert all(item["width"] == TARGET_WIDTH for item in evidence)
        assert all(item["height"] == TARGET_HEIGHT for item in evidence)
        assert all(len(str(item["sha256"])) == 64 for item in evidence)

    assert manager.cleaned
    assert not any((tmp_path / name).exists() for name in manager.names)
