# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the isolated VRAM CLI coordinates existing authorities only."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import measure_anima_regional_lora_vram as cli_module
from tools.anima_regional_lora_performance.isolated_measurement import (
    IsolatedVramResult,
)
from tools.anima_regional_lora_performance.isolated_runner import (
    ANIMA_REGIONAL_LORA_ISOLATED_VRAM_RUNNER,
)
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)


def test_isolated_vram_cli_writes_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep argument, runner, evaluation, and publication coordination thin."""

    manifest = default_scaling_manifest()
    result = IsolatedVramResult((), ())
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "_arguments",
        lambda: type("Args", (), {"output_root": tmp_path})(),
    )
    monkeypatch.setattr(cli_module, "default_scaling_manifest", lambda: manifest)
    monkeypatch.setattr(
        ANIMA_REGIONAL_LORA_ISOLATED_VRAM_RUNNER,
        "run",
        lambda selected: ((), {"device": "test"}),
    )
    monkeypatch.setattr(
        cli_module,
        "evaluate_isolated_vram_result",
        lambda selected, observations: result,
    )

    def write(path: Path, **values: object) -> None:
        """Capture terminal publication and materialize its result path."""

        captured.update(values)
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli_module, "write_isolated_vram_result", write)

    assert cli_module.main() == 0
    assert captured == {
        "manifest": manifest,
        "result": result,
        "environment": {"device": "test"},
    }
    assert tuple(tmp_path.rglob("result.json"))
