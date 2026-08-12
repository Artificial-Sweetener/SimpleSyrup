"""Verify canonical Prompt Control lazy graph expansion observation."""

import pytest

from tools.attention_coupling_benchmark.comfy_probe.prompt_control_expansion import (
    SnapshotPromptControlExpansionV3,
    canonicalize_expansion,
)


def test_canonicalize_expansion_rewrites_created_node_links() -> None:
    """Preserve exact graph inputs while removing execution-specific node IDs."""

    graph = {
        "outer.1": {"class_type": "Encode", "inputs": {"clip": ["source", 0]}},
        "outer.2": {
            "class_type": "Range",
            "inputs": {"conditioning": ["outer.1", 0], "end": 0.5},
        },
    }
    assert canonicalize_expansion(graph) == {
        "node-1": {"class_type": "Encode", "inputs": {"clip": ["source", 0]}},
        "node-2": {
            "class_type": "Range",
            "inputs": {"conditioning": ["node-1", 0], "end": 0.5},
        },
    }


def test_expansion_schema_is_dev_only_and_invalid_values_fail_closed() -> None:
    """Keep expansion observation isolated and JSON-safe."""

    schema = SnapshotPromptControlExpansionV3.define_schema()
    assert schema.node_id == "SimpleSyrupBenchmark.SnapshotPromptControlExpansion"
    assert schema.is_output_node is True
    assert schema.is_dev_only is True
    with pytest.raises(TypeError, match="Unsupported"):
        canonicalize_expansion(
            {"node": {"class_type": "Bad", "inputs": {"value": object()}}}
        )
