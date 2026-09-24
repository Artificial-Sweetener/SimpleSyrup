# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify durable native-Anima comparison evidence persistence."""

from __future__ import annotations

import json
from pathlib import Path

from tools.anima_regional_lora_performance.plain_comparison import (
    write_plain_anima_result,
)


def test_plain_anima_result_is_written_atomically(tmp_path: Path) -> None:
    """Preserve exact result and environment objects without a temporary file."""

    path = tmp_path / "evidence" / "result.json"
    result: dict[str, object] = {
        "profile_id": "plain-anima",
        "median_runtime_ms": 7_500.0,
    }
    environment: dict[str, object] = {"device": "cuda:0"}

    write_plain_anima_result(path, result=result, environment=environment)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "completed",
        "result": result,
        "environment": environment,
    }
    assert not path.with_suffix(".json.tmp").exists()
