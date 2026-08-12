"""Prove exact public graph construction for P9.4 LoRA cases."""

from __future__ import annotations

from tools.anima_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as FULL_NODE_ID,
)
from tools.anima_contextual_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as CONTEXTUAL_NODE_ID,
)
from tools.anima_tiled_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as TILED_NODE_ID,
)
from tools.comfy_api import JsonObject
from tools.text_encoder_lora_integration.fixture import (
    PINNED_TEXT_ENCODER_LORA_FIXTURE,
)
from tools.text_encoder_lora_integration.matrix import cases
from tools.text_encoder_lora_integration.workflow import (
    TextEncoderLoraWorkflowBuilder,
)


def test_workflows_use_explicit_global_and_regional_clip_strengths() -> None:
    """Keep TEXT_ADAPTER model strength zero on both supported encoding paths."""

    definitions = {case.case_id: case for case in cases()}
    builder = TextEncoderLoraWorkflowBuilder()
    global_workflow = builder.build(
        definitions["full-global-text_adapter-text"],
        run_id="global",
        mask_names=("left.png", "right.png"),
    )
    regional_workflow = builder.build(
        definitions["full-regional-text_adapter-text"],
        run_id="regional",
        mask_names=("left.png", "right.png"),
    )
    mixed_workflow = builder.build(
        definitions["full-regional-text_adapter-text-adapter_a-model"],
        run_id="mixed",
        mask_names=("left.png", "right.png"),
    )
    lora_name = PINNED_TEXT_ENCODER_LORA_FIXTURE.lora_name

    global_loader = _single_node(global_workflow.prompt, "LoraLoader")
    assert global_loader["inputs"] == {
        "model": _node_reference(
            global_workflow.prompt, "SimpleSyrup.SimpleLoadAnima", 0
        ),
        "clip": _node_reference(
            global_workflow.prompt, "SimpleSyrup.SimpleLoadAnima", 1
        ),
        "lora_name": lora_name,
        "strength_model": 0.0,
        "strength_clip": 0.75,
    }
    regional_prompt = _positive_prompt(regional_workflow.prompt)
    assert f"<lora:{lora_name}:0:0.75>" in regional_prompt
    assert not _nodes(regional_workflow.prompt, "LoraLoader")
    mixed_prompt = _positive_prompt(mixed_workflow.prompt)
    assert f"<lora:{lora_name}:0:0.75>" in mixed_prompt
    assert "<lora:Anima\\style\\adapter-a.safetensors:0.8:0>" in mixed_prompt


def test_workflows_bind_each_case_to_its_public_spatial_sampler() -> None:
    """Use full 1024 generation and accepted 1024-to-1536 refinement graphs."""

    builder = TextEncoderLoraWorkflowBuilder()
    expected_nodes = {
        "full-baseline": FULL_NODE_ID,
        "tiled-regional-adapter_a-model": TILED_NODE_ID,
        "contextual-regional-adapter_a-model": CONTEXTUAL_NODE_ID,
    }
    definitions = {case.case_id: case for case in cases()}

    for case_id, expected_node in expected_nodes.items():
        workflow = builder.build(
            definitions[case_id],
            run_id=case_id,
            mask_names=("left.png", "right.png"),
        )
        sampler = _single_node(workflow.prompt, expected_node)
        assert expected_node in workflow.required_node_ids
        if expected_node == FULL_NODE_ID:
            assert not _nodes(workflow.prompt, "ImageScale")
            continue
        assert _nodes(workflow.prompt, "ImageScale")
        sampler_inputs = sampler["inputs"]
        assert isinstance(sampler_inputs, dict)
        assert sampler_inputs["diffusion_mode"] == "multidiffusion"


def test_workflows_snapshot_the_exact_public_conditioning_batches() -> None:
    """Bind one output-only snapshot to the unique Schedule-and-Encode outputs."""

    case = next(
        item for item in cases() if item.case_id == "full-regional-text_adapter-text"
    )
    workflow = TextEncoderLoraWorkflowBuilder().build(
        case,
        run_id="snapshot-run",
        mask_names=("left.png", "right.png"),
    )
    encoder_id = next(
        node_id
        for node_id, node in workflow.prompt.items()
        if node.get("class_type")
        == "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl"
    )
    snapshot = _single_node(
        workflow.prompt,
        "SimpleSyrupBenchmark.SnapshotConditioningBatch",
    )

    assert workflow.conditioning_batch_snapshot_node_id is not None
    assert workflow.conditioning_batch_snapshot_run_id == (
        "snapshot-run:full-regional-text_adapter-text:conditioning-batch"
    )
    assert snapshot["inputs"] == {
        "positive": [encoder_id, 1],
        "negative": [encoder_id, 2],
        "run_id": workflow.conditioning_batch_snapshot_run_id,
    }
    assert "SimpleSyrupBenchmark.SnapshotConditioningBatch" in (
        workflow.required_node_ids
    )


def _nodes(prompt: dict[str, JsonObject], class_type: str) -> list[JsonObject]:
    """Return graph nodes with one exact public class identity."""

    return [node for node in prompt.values() if node.get("class_type") == class_type]


def _single_node(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return one uniquely matching graph node."""

    matches = _nodes(prompt, class_type)
    assert len(matches) == 1
    return matches[0]


def _node_reference(
    prompt: dict[str, JsonObject],
    class_type: str,
    output_index: int,
) -> list[object]:
    """Return one graph node's output reference."""

    node_id = next(
        node_id
        for node_id, node in prompt.items()
        if node.get("class_type") == class_type
    )
    return [node_id, output_index]


def _positive_prompt(prompt: dict[str, JsonObject]) -> str:
    """Return the exact lazy regional encoder prompt."""

    node = _single_node(
        prompt,
        "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
    )
    inputs = node["inputs"]
    assert isinstance(inputs, dict)
    value = inputs["positive_prompt"]
    assert isinstance(value, str)
    return value
