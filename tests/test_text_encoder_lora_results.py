"""Prove terminal P9.4 history, pixel, and model-call acceptance."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.anima_attention_coupling_workflow import (
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import ImageReference, JsonObject
from tools.text_encoder_lora_integration.fixture import (
    TextEncoderLoraFixtureIdentity,
)
from tools.text_encoder_lora_integration.matrix import (
    TextEncoderLoraCase,
    TextEncoderLoraSpatialMode,
    cases,
)
from tools.text_encoder_lora_integration.results import (
    TextEncoderLoraResultRecorder,
)


def test_recorder_publishes_complete_changed_pixel_and_call_evidence(
    tmp_path: Path,
) -> None:
    """Require ordered images, exact comparisons, and equal paired call counts."""

    definitions = cases()
    recorder = TextEncoderLoraResultRecorder(tmp_path)
    recorder.record_fixture(_fixture(tmp_path))
    for node_id in ("full", "tiled", "contextual"):
        recorder.record_metadata(node_id, {"display_name": node_id})
    for index, case in enumerate(definitions):
        workflow = _workflow(case.case_id)
        recorder.record_case(
            case,
            workflow,
            prompt_id=f"prompt-{index}",
            history=_history(case, workflow),
            reference=ImageReference(f"{case.case_id}.png", "", "output"),
            image_bytes=_image_bytes(case, index),
            wall_runtime_ms=100.0 + index,
        )

    path = recorder.finalize(
        definitions,
        system_stats={"device": "test"},
        cleanup_verified=True,
        masks_removed=True,
    )
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert len(result["observations"]) == len(definitions)
    assert result["text_encoder_lora_fixture"]["tensor_count"] == 588
    assert set(result["image_comparisons"]) == {
        case.case_id for case in definitions if case.comparison_case_id is not None
    }
    assert all(
        comparison["changed_pixels"] > 0
        for comparison in result["image_comparisons"].values()
    )


def test_recorder_rejects_text_only_hook_in_model_diagnostics(tmp_path: Path) -> None:
    """Fail if a zero-model-strength text LoRA reaches diffusion execution."""

    case = next(
        item for item in cases() if item.case_id == "full-regional-text_adapter-text"
    )
    workflow = _workflow(case.case_id)
    history = _history(case, workflow, force_adapter=True)
    recorder = TextEncoderLoraResultRecorder(tmp_path)

    with pytest.raises(ValueError, match="entered model execution"):
        recorder.record_case(
            case,
            workflow,
            prompt_id="prompt",
            history=history,
            reference=ImageReference("image.png", "", "output"),
            image_bytes=_image_bytes(case, 1),
            wall_runtime_ms=100.0,
        )


def _workflow(case_id: str) -> BuiltAnimaAttentionCouplingWorkflow:
    """Return one minimal typed output-identity fixture."""

    return BuiltAnimaAttentionCouplingWorkflow(
        prompt={"sampler": {"class_type": "fixture", "inputs": {}}},
        save_node_id="save",
        metrics_node_id="metrics",
        metrics_run_id=f"{case_id}:metrics",
        diagnostics_node_id="diagnostics",
        diagnostics_run_id=f"{case_id}:diagnostics",
        conditioning_batch_snapshot_node_id="conditioning-snapshot",
        conditioning_batch_snapshot_run_id=f"{case_id}:conditioning-snapshot",
    )


def _fixture(root: Path) -> TextEncoderLoraFixtureIdentity:
    """Return one already-validated identity for recorder tests."""

    digest = "a" * 64
    return TextEncoderLoraFixtureIdentity(
        lora_name="evidence\\fixture.safetensors",
        path=(root / "fixture.safetensors").resolve(),
        sha256=digest,
        size_bytes=1,
        tensor_count=588,
        source_sha256="b" * 64,
        transformation="fixture",
    )


def _history(
    case: TextEncoderLoraCase,
    workflow: BuiltAnimaAttentionCouplingWorkflow,
    *,
    force_adapter: bool = False,
) -> JsonObject:
    """Return one successful synthetic history with exact P9.4 fields."""

    call_count = {
        TextEncoderLoraSpatialMode.FULL: 12,
        TextEncoderLoraSpatialMode.TILED: 36,
        TextEncoderLoraSpatialMode.CONTEXTUAL: 48,
    }[case.spatial_mode]
    adapter_uses: list[object] = []
    if case.regional_model_lora or force_adapter:
        adapter_uses = [{"target_count": 448, "composition_index": 0}]
    snapshots = [
        {
            "adapter_uses": adapter_uses,
            "regional_conditioning_entries": [{"region_index": 0}, {"region_index": 1}],
        }
    ]
    return {
        "status": {"status_str": "success"},
        "outputs": {
            workflow.metrics_node_id: {
                "benchmark_metrics": [
                    {
                        "run_id": workflow.metrics_run_id,
                        "model_call_count": call_count,
                        "runtime_ms": 100.0,
                        "peak_vram_bytes": 1024,
                    }
                ]
            },
            workflow.diagnostics_node_id: {
                "regional_diagnostics": [
                    {
                        "run_id": workflow.diagnostics_run_id,
                        "record_count": len(snapshots),
                        "snapshots": snapshots,
                    }
                ]
            },
            workflow.conditioning_batch_snapshot_node_id: {
                "conditioning_batch_snapshot": [
                    {
                        "run_id": workflow.conditioning_batch_snapshot_run_id,
                        "positive": {
                            "kind": "conditioning_batch",
                            "batch_entries": [],
                        },
                        "negative": {
                            "kind": "conditioning_batch",
                            "batch_entries": [],
                        },
                    }
                ]
            },
        },
    }


def _image_bytes(case: TextEncoderLoraCase, index: int) -> bytes:
    """Return one valid labeled-size PNG with case-distinct pixels."""

    size = 1024 if case.spatial_mode is TextEncoderLoraSpatialMode.FULL else 1536
    image = Image.new("RGB", (size, size), (index % 256, 40, 80))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
