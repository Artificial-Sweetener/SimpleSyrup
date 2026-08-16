# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the scaling CLI publishes its complete evaluated result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import benchmark_anima_regional_lora_scaling as cli_module
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_results import (
    ScalingPerformanceResult,
    ScalingProfileResult,
)
from tools.anima_regional_lora_performance.matrix_runner import (
    ANIMA_REGIONAL_LORA_SCALING_RUNNER,
)


def test_scaling_cli_writes_result_and_returns_gate_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep command-line coordination thin over runner/result authorities."""

    manifest = default_scaling_manifest()
    result = ScalingPerformanceResult(
        (),
        tuple(
            ScalingProfileResult(
                profile.profile_id,
                1.0,
                1,
                None,
                profile.maximum_overhead_percent,
                profile.work.prepared_cache_entries,
                0,
                profile.equality_profile_id,
                0.0,
                0.0,
                True,
                True,
                True,
                True,
                True,
            )
            for profile in manifest.profiles
        ),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "_arguments",
        lambda: type(
            "Args",
            (),
            {
                "repeats": 3,
                "output_root": tmp_path,
                "artifact_inventory": tmp_path / "inventory.json",
            },
        )(),
    )
    monkeypatch.setattr(cli_module, "load_performance_artifacts", lambda path: ())
    monkeypatch.setattr(
        cli_module,
        "default_scaling_manifest",
        lambda *, repeats, artifacts: manifest,
    )
    monkeypatch.setattr(
        ANIMA_REGIONAL_LORA_SCALING_RUNNER,
        "run",
        lambda selected: ((), {"device": "test"}),
    )
    monkeypatch.setattr(
        cli_module,
        "evaluate_scaling_result",
        lambda selected, observations: result,
    )

    def write(path: Path, **values: object) -> None:
        """Capture publication arguments and materialize recognizable JSON."""

        captured.update(values)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"passed": True}), encoding="utf-8")

    monkeypatch.setattr(cli_module, "write_scaling_result", write)

    assert cli_module.main() == 0
    assert captured["manifest"] is manifest
    assert captured["result"] is result
    assert captured["environment"] == {"device": "test"}
    assert tuple(tmp_path.rglob("result.json"))
