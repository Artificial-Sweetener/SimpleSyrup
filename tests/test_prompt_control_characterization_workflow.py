"""Verify public-node Prompt Control characterization graph construction."""

from tools.prompt_control_characterization.cases import cases
from tools.prompt_control_characterization.workflow import (
    PromptControlWorkflowBuilder,
)


def test_every_case_uses_public_prompt_control_and_runtime_probe_nodes() -> None:
    """Compose real lazy expansion and sampling without parser duplication."""

    builder = PromptControlWorkflowBuilder()
    for case in cases():
        built = builder.build(case)
        node_types = [node["class_type"] for node in built.prompt.values()]
        assert "SimpleSyrupBenchmark.SnapshotPromptControlExpansion" in node_types
        assert "SimpleSyrupBenchmark.SnapshotPromptControl" in node_types
        assert "SimpleSyrupBenchmark.InstrumentPromptControlModel" in node_types
        assert "SimpleSyrupBenchmark.ReadPromptControlRuntime" in node_types
        assert "KSampler" in node_types
        assert not any("SaveImage" == node_type for node_type in node_types)
        if case.lora_text:
            assert "PCLazyLoraLoaderAdvanced" in node_types


def test_text_shapes_use_prompt_control_ranges_and_lazy_expansion() -> None:
    """Keep adjacent lazy expansion separate from explicit overlap ranges."""

    builder = PromptControlWorkflowBuilder()
    by_id = {case.case_id: case for case in cases()}
    adjacent = builder.build(by_id["text-adjacent"]).prompt
    overlap = builder.build(by_id["text-overlapping"]).prompt
    adjacent_types = [node["class_type"] for node in adjacent.values()]
    overlap_types = [node["class_type"] for node in overlap.values()]
    assert adjacent_types.count("PCLazyTextEncode") == 1
    assert overlap_types.count("PCTextEncodeWithRange") == 3
    assert overlap_types.count("ConditioningCombine") == 1
