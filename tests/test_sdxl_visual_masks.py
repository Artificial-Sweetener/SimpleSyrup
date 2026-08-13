# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify deterministic U11 hard, overlap, and uncovered mask profiles."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from PIL import Image

from tools.sdxl_attention_coupling_integration.visual_cases import VisualMaskProfile
from tools.sdxl_attention_coupling_integration.visual_masks import (
    ManagedSdxlVisualMasks,
)


def test_mask_profiles_have_declared_overlap_and_gap(tmp_path: Path) -> None:
    """Keep hard, soft-overlap, and uncovered-center geometry distinguishable."""

    owner = ManagedSdxlVisualMasks(input_root=tmp_path, run_id="run")

    with owner:
        hard = _columns(tmp_path, owner.names(VisualMaskProfile.HARD))
        overlap = _columns(tmp_path, owner.names(VisualMaskProfile.SOFT_OVERLAP))
        gap = _columns(tmp_path, owner.names(VisualMaskProfile.UNCOVERED_CENTER))
        assert hard[0] & hard[1]
        assert len(overlap[0] & overlap[1]) > len(hard[0] & hard[1])
        assert not gap[0] & gap[1]
        assert max(gap[0]) < min(gap[1]) - 1
        evidence = owner.evidence()
        assert len(evidence) == 6
        assert all(
            item["width"] == 1536 and item["height"] == 1536 for item in evidence
        )

    assert owner.cleaned
    assert not list(tmp_path.glob("simple_syrup_u11_*.png"))


def _columns(root: Path, names: tuple[str, str]) -> tuple[set[int], set[int]]:
    """Return active center-row columns for one ordered mask pair."""

    result: list[set[int]] = []
    for name in names:
        with Image.open(root / name) as image:
            grayscale = image.convert("L")
            result.append(
                {
                    x
                    for x in range(grayscale.width)
                    if cast(int, grayscale.getpixel((x, 768))) > 0
                }
            )
    return result[0], result[1]
