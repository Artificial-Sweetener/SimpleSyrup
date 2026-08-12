"""Verify the P9.1 managed coordinator remains a thin explicit entrypoint."""

from pathlib import Path
from subprocess import run


def test_cli_help_exposes_managed_source_baseline_and_timeout_boundaries() -> None:
    """Keep external install and evidence locations caller-controlled."""

    python = Path(r"<COMFY_ROOT>\venv\Scripts\python.exe")
    completed = run(
        [
            str(python),
            "-m",
            "tools.run_prompt_control_attention_coupling_integration",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--comfy-root" in completed.stdout
    assert "--prompt-control-root" in completed.stdout
    assert "--baseline-path" in completed.stdout
    assert "--output-root" in completed.stdout
    assert "--prompt-timeout" in completed.stdout
