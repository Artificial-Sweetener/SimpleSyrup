"""Verify exact P9.1 history output decoding."""

from __future__ import annotations

import pytest

from tools.prompt_control_attention_coupling_integration.history import parse_outputs
from tools.prompt_control_attention_coupling_integration.matrix import cases
from tools.prompt_control_attention_coupling_integration.workflow import (
    PromptControlAttentionWorkflowBuilder,
)


def test_history_parser_reads_exact_conditioning_metrics_and_diagnostics() -> None:
    """Require one record from every conditioning and sampler evidence owner."""

    workflow = PromptControlAttentionWorkflowBuilder().build(
        cases()[0],
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    expansion_id = workflow.conditioning_evidence.expansion_node_id
    snapshot_id = workflow.conditioning_evidence.snapshot_node_id
    assert expansion_id is not None
    assert snapshot_id is not None
    history = {
        "status": {"status_str": "success", "messages": []},
        "outputs": {
            expansion_id: {"prompt_control_expansion": [{"case_id": "case"}]},
            snapshot_id: {"prompt_control_snapshot": [{"case_id": "case"}]},
            workflow.metrics_node_id: {"benchmark_metrics": [{"run_id": "metrics"}]},
            workflow.diagnostics_node_id: {
                "regional_diagnostics": [{"run_id": "diagnostics"}]
            },
        },
    }

    outputs = parse_outputs(history, workflow)

    assert outputs.expansion == {"case_id": "case"}
    assert outputs.snapshot == {"case_id": "case"}
    assert outputs.metrics == {"run_id": "metrics"}
    assert outputs.diagnostics == {"run_id": "diagnostics"}


def test_history_parser_rejects_failed_or_incomplete_evidence() -> None:
    """Fail closed before accepting partial host execution."""

    workflow = PromptControlAttentionWorkflowBuilder().build(
        cases()[0],
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    with pytest.raises(RuntimeError, match="Comfy execution failed"):
        parse_outputs(
            {"status": {"status_str": "error", "messages": ["failure"]}},
            workflow,
        )
