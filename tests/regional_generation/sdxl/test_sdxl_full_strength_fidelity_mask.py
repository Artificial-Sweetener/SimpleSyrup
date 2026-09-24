# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify full-frame regional fidelity mask ownership and cleanup."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.sdxl_full_strength_lora_fidelity.mask import ManagedAllOneMask


def test_all_one_mask_is_exact_and_cleanup_owned(tmp_path: Path) -> None:
    """Write one all-white 1024-square mask and remove only that file."""

    with ManagedAllOneMask(input_root=tmp_path, run_id="proof") as mask:
        path = tmp_path / mask.name
        with Image.open(path) as image:
            assert image.size == (1024, 1024)
            assert image.mode == "L"
            assert image.getextrema() == (255, 255)

    assert mask.cleaned
    assert not path.exists()
