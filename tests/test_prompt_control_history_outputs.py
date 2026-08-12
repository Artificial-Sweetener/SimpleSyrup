"""Verify strict decoding of Prompt Control Comfy history outputs."""

import pytest

from tools.prompt_control_characterization.history_outputs import parse_outputs


def test_parse_outputs_reads_exact_snapshot_and_runtime_records() -> None:
    """Decode only successful histories with one record from each probe."""

    history = {
        "status": {"status_str": "success"},
        "outputs": {
            "9": {"prompt_control_expansion": [{"case_id": "case"}]},
            "10": {"prompt_control_snapshot": [{"case_id": "case"}]},
            "11": {"prompt_control_runtime": [{"run_id": "case"}]},
        },
    }
    outputs = parse_outputs(
        history,
        expansion_node_id="9",
        snapshot_node_id="10",
        runtime_node_id="11",
    )
    assert outputs.expansion["case_id"] == "case"
    assert outputs.snapshot["case_id"] == "case"
    assert outputs.runtime["run_id"] == "case"


def test_parse_outputs_rejects_failed_history() -> None:
    """Surface Comfy execution failure instead of accepting partial evidence."""

    with pytest.raises(RuntimeError, match="failed"):
        parse_outputs(
            {"status": {"status_str": "error", "messages": ["failed"]}},
            expansion_node_id="0",
            snapshot_node_id="1",
            runtime_node_id="2",
        )
