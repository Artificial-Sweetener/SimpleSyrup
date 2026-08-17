# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the Anima performance benchmark's host-runtime CLI boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_benchmark_supports_direct_filename_execution(tmp_path: Path) -> None:
    """Resolve repository and Comfy packages outside either source root."""

    script = Path(__file__).parents[1] / "tools" / "benchmark_anima_regional_lora.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Run the pinned full-context Anima" in result.stdout
