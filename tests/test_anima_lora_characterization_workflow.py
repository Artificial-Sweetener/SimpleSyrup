"""Verify ordinary global Comfy graphs for pinned ADAPTER_A characterization."""

from __future__ import annotations

import json
from typing import TypedDict, cast

from tools.anima_lora_characterization.matrix import LoraRun, profiles
from tools.anima_lora_characterization.workflow import LoraWorkflowBuilder
from tools.attention_coupling_benchmark.manifest import load_manifest


class _Node(TypedDict):
    """Narrow one generated node for contract assertions."""

    class_type: str
    inputs: dict[str, object]


def test_static_four_adapter_graph_preserves_loader_order() -> None:
    """Chain four ordinary model-only loaders before instrumentation."""

    profile = next(item for item in profiles() if item.profile_id == "static-4")
    run = LoraRun(profile, 1_029_384_756, capture_outputs=False)
    built = LoraWorkflowBuilder().build(run, load_manifest().sampling, "test prompt")
    loaders = _nodes(built.prompt, "LoraLoaderModelOnly")

    assert [node["inputs"]["strength_model"] for node in loaders] == [
        0.1,
        0.2,
        0.3,
        0.4,
    ]
    instrument = _nodes(built.prompt, "SimpleSyrupBenchmark.InstrumentLoraModel")[0]
    assert json.loads(cast(str, instrument["inputs"]["adapter_identities_json"])) == [
        adapter.identity for adapter in profile.adapters
    ]
    assert instrument["inputs"]["capture_outputs"] is False
    assert not _nodes(built.prompt, "CreateHookLoraModelOnly")


def test_scheduled_four_adapter_graph_keeps_independent_boundaries() -> None:
    """Build and order four independent WeightHooks before conditioning attach."""

    profile = next(item for item in profiles() if item.profile_id == "scheduled-4")
    run = LoraRun(profile, 1_029_384_756, capture_outputs=True)
    built = LoraWorkflowBuilder().build(run, load_manifest().sampling, "test prompt")

    assert len(_nodes(built.prompt, "CreateHookLoraModelOnly")) == 4
    assert len(_nodes(built.prompt, "CreateHookKeyframe")) == 16
    assert len(_nodes(built.prompt, "SetHookKeyframes")) == 4
    assert len(_nodes(built.prompt, "CombineHooks4")) == 1
    assert len(_nodes(built.prompt, "PairConditioningSetProperties")) == 1
    assert not _nodes(built.prompt, "LoraLoaderModelOnly")
    keyframes = _nodes(built.prompt, "CreateHookKeyframe")
    assert [node["inputs"]["start_percent"] for node in keyframes[:4]] == [
        0.0,
        0.25,
        0.5,
        0.75,
    ]
    assert [node["inputs"]["strength_mult"] for node in keyframes[:4]] == [
        0.0,
        1.0,
        0.5,
        0.0,
    ]


def _nodes(prompt: dict[str, dict[str, object]], class_type: str) -> list[_Node]:
    """Return graph nodes with narrowed input dictionaries."""

    result: list[_Node] = []
    for node in prompt.values():
        if node.get("class_type") != class_type:
            continue
        inputs = node.get("inputs")
        assert isinstance(inputs, dict)
        assert all(isinstance(key, str) for key in inputs)
        result.append(
            {"class_type": class_type, "inputs": cast(dict[str, object], inputs)}
        )
    return result
