# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the Anima oracle's direct command-line boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tools.run_anima_regression_oracle as oracle
from tools.anima_regression_oracle.manifest import OracleCommand


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


def test_complete_level_includes_every_declared_managed_rerun() -> None:
    """Require the complete oracle to execute runtime evidence after code gates."""

    focused = OracleCommand("focused", ("focused",))
    repository = (OracleCommand("repository", ("repository",)),)
    managed = (OracleCommand("managed", ("managed",)),)

    commands = oracle._commands("complete", focused, repository, managed)

    assert [command.identity for command in commands] == [
        "focused",
        "repository",
        "managed",
    ]
