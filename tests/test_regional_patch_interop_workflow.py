"""Verify the closed P9.7 public modifier workflow matrix."""

from __future__ import annotations

import pytest

from tools.regional_patch_interop_integration.matrix import (
    PatchInteropModelFamily,
    PatchInteropModifier,
    PatchInteropSpatialMode,
    RegionalPatchInteropCase,
    cases,
)
from tools.regional_patch_interop_integration.workflow import (
    CONTEXTUAL_NODE_ID,
    FULL_NODE_ID,
    TILED_NODE_ID,
    RegionalPatchInteropWorkflowBuilder,
)


def test_matrix_contains_five_acceptances_and_six_exact_rejections() -> None:
    """Keep every required modifier, spatial, scheduled, and family case."""

    definitions = cases()

    assert len(definitions) == 11
    assert sum(case.expect_success for case in definitions) == 5
    assert {case.modifier for case in definitions} == set(PatchInteropModifier)
    assert {case.spatial_mode for case in definitions} == set(PatchInteropSpatialMode)
    assert {case.model_family for case in definitions} == set(PatchInteropModelFamily)


@pytest.mark.parametrize(
    "case",
    tuple(
        case for case in cases() if case.model_family is PatchInteropModelFamily.ANIMA
    ),
    ids=lambda case: case.case_id,
)
def test_anima_workflows_route_modifier_snapshot_and_target_sampler(
    case: RegionalPatchInteropCase,
) -> None:
    """Keep modifier, observer, conditioning, and sampler ownership sequential."""

    definition = next(item for item in cases() if item.case_id == case.case_id)
    workflow = RegionalPatchInteropWorkflowBuilder().build(
        definition,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    snapshot = workflow.prompt[workflow.modifier_snapshot_node_id]
    snapshot_inputs = snapshot["inputs"]
    assert isinstance(snapshot_inputs, dict)
    expected_modifier_node = {
        PatchInteropModifier.NONE: "SimpleSyrup.SimpleLoadAnima",
        PatchInteropModifier.EASYCACHE: "EasyCache",
        PatchInteropModifier.LAZYCACHE: "LazyCache",
        PatchInteropModifier.OPTIMIZED_ATTENTION: "ModelAttentionSelector",
        PatchInteropModifier.NEGPIP: "CLIPNegPip",
    }[definition.modifier]
    modifier_reference = snapshot_inputs["model"]
    assert isinstance(modifier_reference, list)
    modifier_node = workflow.prompt[str(modifier_reference[0])]
    assert modifier_node["class_type"] == expected_modifier_node
    assert (
        workflow.prompt[workflow.sampler_node_id]["class_type"]
        == {
            PatchInteropSpatialMode.FULL: FULL_NODE_ID,
            PatchInteropSpatialMode.TILED: TILED_NODE_ID,
            PatchInteropSpatialMode.CONTEXTUAL: CONTEXTUAL_NODE_ID,
        }[definition.spatial_mode]
    )


@pytest.mark.parametrize(
    "case_id",
    [
        "anima-tiled-optimized-attention",
        "anima-contextual-optimized-attention",
    ],
)
def test_accepted_spatial_cases_generate_a_real_1024_source_before_1536_refinement(
    case_id: str,
) -> None:
    """Reserve tiled and Contextual success for the approved upscale flow."""

    definition = next(case for case in cases() if case.case_id == case_id)
    workflow = RegionalPatchInteropWorkflowBuilder().build(
        definition,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )

    assert _node_count(workflow.prompt, FULL_NODE_ID) == 1
    assert workflow.source_save_node_id is not None
    scale = _only_node(workflow.prompt, "ImageScale")
    scale_inputs = _inputs(scale)
    assert scale_inputs["width"] == 1536
    assert scale_inputs["height"] == 1536


@pytest.mark.parametrize(
    "case_id",
    ["anima-tiled-easycache-rejected", "anima-contextual-easycache-rejected"],
)
def test_rejected_spatial_cases_use_unsampled_1024_to_1536_latent(case_id: str) -> None:
    """Prove spatial policy without a denoiser call before target rejection."""

    definition = next(case for case in cases() if case.case_id == case_id)
    workflow = RegionalPatchInteropWorkflowBuilder().build(
        definition,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )

    assert _node_count(workflow.prompt, FULL_NODE_ID) == 0
    assert workflow.source_save_node_id is None
    source = _only_node(workflow.prompt, "EmptyCosmosLatentVideo")
    source_inputs = _inputs(source)
    assert source_inputs["width"] == 1024
    assert source_inputs["height"] == 1024
    scale = _only_node(workflow.prompt, "ImageScale")
    scale_inputs = _inputs(scale)
    assert scale_inputs["width"] == 1536
    assert scale_inputs["height"] == 1536


def test_scheduled_cache_graph_authors_the_exact_regional_adapter_a_interval() -> None:
    """Reach cache rejection through public scheduled Prompt Control metadata."""

    definition = next(
        case
        for case in cases()
        if case.case_id == "anima-full-scheduled-easycache-rejected"
    )
    workflow = RegionalPatchInteropWorkflowBuilder().build(
        definition,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    encoded = _only_node(
        workflow.prompt,
        "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
    )

    assert "[<lora:Anima\\style\\adapter-a.safetensors:0.75>:0.25,0.75]" in str(
        _inputs(encoded)["positive_prompt"]
    )


def test_sdxl_negpip_graph_uses_public_modifier_snapshot_and_sampler() -> None:
    """Submit SDXL NegPiP through the same evidence and rejection boundary."""

    definition = next(
        case for case in cases() if case.case_id == "sdxl-negpip-rejected"
    )
    workflow = RegionalPatchInteropWorkflowBuilder().build(
        definition,
        run_id="run",
        mask_names=("left.png", "right.png"),
        sdxl_checkpoint_name="managed.safetensors",
    )

    assert _node_count(workflow.prompt, "CheckpointLoaderSimple") == 1
    assert _node_count(workflow.prompt, "CLIPNegPip") == 1
    assert (
        _node_count(
            workflow.prompt,
            "SimpleSyrupBenchmark.SnapshotModelModifier",
        )
        == 1
    )
    assert workflow.prompt[workflow.sampler_node_id]["class_type"] == FULL_NODE_ID


def _node_count(prompt: dict[str, dict[str, object]], class_type: str) -> int:
    """Return one graph class cardinality."""

    return sum(node["class_type"] == class_type for node in prompt.values())


def _only_node(
    prompt: dict[str, dict[str, object]],
    class_type: str,
) -> dict[str, object]:
    """Return the unique graph node for one class."""

    nodes = tuple(node for node in prompt.values() if node["class_type"] == class_type)
    assert len(nodes) == 1
    return nodes[0]


def _inputs(node: dict[str, object]) -> dict[str, object]:
    """Return one narrowed graph input dictionary."""

    value = node["inputs"]
    assert isinstance(value, dict)
    return value
