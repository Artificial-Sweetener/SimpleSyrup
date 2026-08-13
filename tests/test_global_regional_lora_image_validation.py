# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify P9.3 same-server transition image acceptance."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tools.global_regional_lora_integration.image_validation import (
    GlobalRegionalLoraImageValidator,
)


def test_accepts_repeatable_distinct_global_effect(tmp_path: Path) -> None:
    """Require equal repeated composition and different regional-only pixels."""

    images = _images(
        tmp_path,
        before=(200, 20, 30),
        after=(200, 20, 30),
        regional=(20, 30, 200),
    )

    evidence = GlobalRegionalLoraImageValidator().validate(images)

    assert evidence.distinct_before_after.changed_pixels == 0
    assert evidence.distinct_before_regional_only.changed_pixels == 2
    assert evidence.distinct_before_regional_only.maximum_channel_delta == 180


def test_rejects_changed_repeated_composition(tmp_path: Path) -> None:
    """Fail when a same-process transition changes identical sampling inputs."""

    images = _images(
        tmp_path,
        before=(200, 20, 30),
        after=(199, 20, 30),
        regional=(20, 30, 200),
    )

    with pytest.raises(ValueError, match="changed across"):
        GlobalRegionalLoraImageValidator().validate(images)


def test_rejects_absent_distinct_global_effect(tmp_path: Path) -> None:
    """Fail when distinct global/regional pixels equal regional-only output."""

    images = _images(
        tmp_path,
        before=(20, 30, 200),
        after=(20, 30, 200),
        regional=(20, 30, 200),
    )

    with pytest.raises(ValueError, match="no decoded pixel effect"):
        GlobalRegionalLoraImageValidator().validate(images)


def _images(
    root: Path,
    *,
    before: tuple[int, int, int],
    after: tuple[int, int, int],
    regional: tuple[int, int, int],
) -> dict[str, Path]:
    """Write one complete transition fixture and return its labeled paths."""

    colors = {
        "global-global_adapter-regional-adapter_a-before": before,
        "global-global_adapter-regional-adapter_a-after": after,
        "regional-adapter_a-only": regional,
    }
    paths: dict[str, Path] = {}
    for case_id, color in colors.items():
        path = root / f"{case_id}.png"
        Image.new("RGB", (2, 1), color).save(path)
        paths[case_id] = path
    return paths
