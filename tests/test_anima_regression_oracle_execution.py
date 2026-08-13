# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify ordered fail-fast Anima oracle command execution."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.anima_regression_oracle.execution import OracleCommandExecutor
from tools.anima_regression_oracle.manifest import OracleCommand


def test_executor_preserves_logs_and_stops_after_first_failure(tmp_path: Path) -> None:
    """Retain the failed gate without running later work."""

    commands = (
        OracleCommand(
            "passes",
            (sys.executable, "-c", "print('passed output')"),
        ),
        OracleCommand(
            "fails",
            (
                sys.executable,
                "-c",
                "import sys; print('failed output', file=sys.stderr); sys.exit(7)",
            ),
        ),
        OracleCommand("must-not-run", (sys.executable, "-c", "print('wrong')")),
    )

    observations = OracleCommandExecutor().execute(
        commands,
        repo_root=Path.cwd(),
        output_root=tmp_path,
    )

    assert [item.identity for item in observations] == ["passes", "fails"]
    assert [item.return_code for item in observations] == [0, 7]
    assert (tmp_path / observations[0].stdout_file).read_text(
        encoding="utf-8"
    ).strip() == "passed output"
    assert (tmp_path / observations[1].stderr_file).read_text(
        encoding="utf-8"
    ).strip() == "failed output"
    assert not any(tmp_path.glob("*must-not-run*"))
