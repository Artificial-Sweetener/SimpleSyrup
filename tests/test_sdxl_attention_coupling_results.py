"""Verify complete SDXL history decoding and acceptance persistence."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.history import decode_sdxl_history
from tools.sdxl_attention_coupling_integration.matrix import MODES
from tools.sdxl_attention_coupling_integration.results import (
    SdxlIntegrationResultRecorder,
)
from tools.sdxl_attention_coupling_integration.workflow import (
    BuiltSdxlAttentionCouplingWorkflow,
    build_sdxl_attention_coupling_workflow,
)


def test_result_requires_exact_calls_diagnostics_dimensions_and_cleanup(
    tmp_path: Path,
) -> None:
    """Persist all three labeled outputs only after direct acceptance evidence."""

    workflow = build_sdxl_attention_coupling_workflow(
        run_id="run-1",
        checkpoint_name="owned\\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
    )
    history = _history(workflow)
    evidence = decode_sdxl_history(history, workflow)
    images = {
        mode.mode_id: _png(mode.width, mode.height, index)
        for index, mode in enumerate(MODES, 1)
    }
    recorder = SdxlIntegrationResultRecorder(tmp_path)

    recorder.record_workflow(
        workflow,
        history=history,
        prompt_id="prompt-1",
        evidence=evidence,
        image_bytes=images,
        masks=({"name": "left.png"}, {"name": "right.png"}),
        wall_runtime_ms=100.0,
    )
    result_path = recorder.finalize(
        system_stats={"system": "test"},
        server_cleanup=True,
        checkpoint_cleanup=True,
        mask_cleanup=True,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert [item["mode_id"] for item in result["observations"]] == [
        mode.mode_id for mode in MODES
    ]
    assert [item["metrics"]["model_call_count"] for item in result["observations"]] == [
        mode.expected_model_calls for mode in MODES
    ]
    assert result["upscale_factor"] == 1.5


def _history(workflow: BuiltSdxlAttentionCouplingWorkflow) -> JsonObject:
    """Build one complete managed-history payload from workflow output IDs."""

    outputs: JsonObject = {}
    for mode in MODES:
        terminal = workflow.outputs[mode.mode_id]
        snapshots = [
            {
                "strategy": "attention_coupling",
                "backend": "comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "active_region_indices": [0, 1],
                "estimated_work": {
                    "cross_attention_branch_multiplier": 3.0,
                    "denoiser_call_multiplier": 1.0,
                },
                "spatial_mode": spatial_mode,
            }
            for spatial_mode in sorted(mode.expected_spatial_modes)
        ]
        outputs[terminal.save_node_id] = {
            "images": [
                {
                    "filename": f"{mode.mode_id}.png",
                    "subfolder": "p8.5",
                    "type": "output",
                }
            ]
        }
        outputs[terminal.metrics_node_id] = {
            "benchmark_metrics": [
                {
                    "model_call_count": mode.expected_model_calls,
                    "runtime_ms": 10.0,
                    "peak_vram_bytes": 1,
                }
            ]
        }
        outputs[terminal.diagnostics_node_id] = {
            "regional_diagnostics": [
                {"record_count": len(snapshots), "snapshots": snapshots}
            ]
        }
    return {
        "status": {"completed": True, "status_str": "success", "messages": []},
        "outputs": outputs,
    }


def _png(width: int, height: int, value: int) -> bytes:
    """Return one nonconstant RGB PNG at exact output dimensions."""

    image = Image.new("RGB", (width, height), (value, value, value))
    image.putpixel((0, 0), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
