# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete P10.2 result persistence and acceptance."""

import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from tools.comfy_api import ImageReference, JsonObject
from tools.regional_strategy_comparison.history import (
    CompletedStrategyEvidence,
    StrategyProbeMetrics,
)
from tools.regional_strategy_comparison.matrix import (
    cases,
    expected_low_rank_multiplier,
)
from tools.regional_strategy_comparison.results import StrategyComparisonResultRecorder
from tools.regional_strategy_comparison.workflow import (
    StrategyComparisonWorkflowBuilder,
)


def _png(width: int, height: int) -> bytes:
    """Return one deterministic valid image artifact."""

    stream = BytesIO()
    Image.new("RGB", (width, height), color=(90, 70, 50)).save(stream, format="PNG")
    return stream.getvalue()


def test_result_accepts_complete_call_and_trajectory_matrix(tmp_path: Path) -> None:
    """Persist all labeled rows and validate shared-source identity."""

    root = tmp_path
    recorder = StrategyComparisonResultRecorder(root)
    recorder.record_metadata({"node": {"name": "sampler"}})
    definitions = cases()
    source = _png(1024, 1024)
    (root / "shared-source.png").write_bytes(source)
    builder = StrategyComparisonWorkflowBuilder()
    for case in definitions:
        attention = case.strategy == "attention_coupling"
        sizes = (1, 1) if attention else (2, 2)
        metrics = StrategyProbeMetrics(5.0, 100, 2, sizes, sum(sizes))
        diagnostics: tuple[JsonObject, ...] = ()
        if attention:
            work: JsonObject = {
                "denoiser_call_multiplier": 1.0,
                "cross_attention_branch_multiplier": 3.0,
            }
            low_rank = expected_low_rank_multiplier(case)
            if low_rank:
                work["low_rank_adapter_multiplier"] = low_rank
            snapshot: JsonObject = {
                "strategy": "attention_coupling",
                "spatial_mode": "full" if not case.is_refinement else "tile",
                "estimated_work": work,
            }
            diagnostics = (snapshot, snapshot)
        evidence = CompletedStrategyEvidence(
            metrics,
            diagnostics,
            ImageReference("image.png", "p10.2", "output"),
        )
        workflow = builder.build(
            case,
            run_id="run-1",
            mask_names=("left.png", "right.png"),
            shared_source_name="source.png",
        )
        image = source if not case.is_refinement else _png(1536, 1536)
        recorder.record_case(
            case,
            workflow,
            evidence,
            history={},
            prompt_id=f"prompt-{case.case_id}",
            image_bytes=image,
            wall_runtime_ms=10.0,
        )

    result = recorder.finalize(
        definitions,
        source_sha256=hashlib.sha256(source).hexdigest(),
        system_stats={},
        cleanup_verified=True,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert len(payload["observations"]) == 16
    assert payload["cleanup_verified"] is True
