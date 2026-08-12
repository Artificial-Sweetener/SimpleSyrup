"""Verify typed P10.2 history extraction."""

from tools.comfy_api import JsonObject
from tools.regional_strategy_comparison.history import parse_completed_evidence
from tools.regional_strategy_comparison.matrix import cases
from tools.regional_strategy_comparison.workflow import (
    StrategyComparisonWorkflowBuilder,
)


def test_history_extracts_probe_batches_diagnostics_and_image() -> None:
    """Narrow every new probe field and the exact diagnostic identity."""

    case = cases()[0]
    workflow = StrategyComparisonWorkflowBuilder().build(
        case,
        run_id="run-1",
        mask_names=("left.png", "right.png"),
        shared_source_name="source.png",
    )
    assert workflow.diagnostics_node_id is not None
    snapshot = {
        "strategy": "attention_coupling",
        "estimated_work": {
            "denoiser_call_multiplier": 1.0,
            "cross_attention_branch_multiplier": 3.0,
        },
    }
    history: JsonObject = {
        "status": {"status_str": "success", "completed": True},
        "outputs": {
            workflow.metrics_node_id: {
                "benchmark_metrics": [
                    {
                        "run_id": workflow.metrics_run_id,
                        "runtime_ms": 12.5,
                        "peak_vram_bytes": 100,
                        "model_call_count": 2,
                        "model_input_batch_sizes": [1, 3],
                        "model_input_batch_elements": 4,
                    }
                ]
            },
            workflow.diagnostics_node_id: {
                "regional_diagnostics": [
                    {
                        "run_id": workflow.diagnostics_run_id,
                        "record_count": 1,
                        "snapshots": [snapshot],
                    }
                ]
            },
            workflow.save_node_id: {
                "images": [
                    {"filename": "out.png", "subfolder": "p10.2", "type": "output"}
                ]
            },
        },
    }

    evidence = parse_completed_evidence(history, workflow)

    assert evidence.metrics.model_input_batch_sizes == (1, 3)
    assert evidence.metrics.model_input_batch_elements == 4
    assert evidence.diagnostics == (snapshot,)
    assert evidence.image_reference.filename == "out.png"
