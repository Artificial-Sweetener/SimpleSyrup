# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the Prompt Control coordinator remains a thin module entrypoint."""

from pathlib import Path
from subprocess import run


def test_cli_help_exposes_explicit_external_boundaries() -> None:
    """Keep source root, loopback server, output, and timeout caller-controlled."""

    python = Path(r"<COMFY_ROOT>\venv\Scripts\python.exe")
    completed = run(
        [str(python), "-m", "tools.characterize_prompt_control", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--prompt-control-root" in completed.stdout
    assert "--server-url" in completed.stdout
    assert "--output-root" in completed.stdout
    assert "--prompt-timeout" in completed.stdout
