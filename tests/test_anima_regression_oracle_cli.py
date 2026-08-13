# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the Anima oracle's direct command-line boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_oracle_script_supports_direct_filename_execution(tmp_path: Path) -> None:
    """Load repository-owned packages even when launched outside the repository."""

    script = Path(__file__).parents[1] / "tools" / "run_anima_regression_oracle.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Run the blocking accepted Anima regression oracle" in result.stdout
