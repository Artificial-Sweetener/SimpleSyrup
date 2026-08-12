"""Verify public graph construction for P10.2 comparison cases."""

from tools.regional_strategy_comparison.matrix import cases
from tools.regional_strategy_comparison.workflow import (
    StrategyComparisonWorkflowBuilder,
)


def test_workflows_route_each_strategy_and_spatial_profile() -> None:
    """Bind every matrix row to its exact public sampler and evidence nodes."""

    builder = StrategyComparisonWorkflowBuilder()
    for case in cases():
        built = builder.build(
            case,
            run_id="run-1",
            mask_names=("left.png", "right.png"),
            shared_source_name="source.png",
        )
        sampler = next(
            node
            for node in built.prompt.values()
            if str(node["class_type"]).startswith("SimpleSyrup.KSampler")
        )
        class_type = str(sampler["class_type"])
        assert ("AttentionCoupling" in class_type) == (
            case.strategy == "attention_coupling"
        )
        assert ("Tiled" in class_type or "TiledRegion" in class_type) == (
            case.spatial_profile.startswith("tiled_")
        )
        assert ("Contextual" in class_type) == (
            case.spatial_profile.startswith("contextual_")
        )
        assert (built.diagnostics_node_id is not None) == (
            case.strategy == "attention_coupling"
        )


def test_refinement_workflows_load_one_shared_source_at_exact_1_5x() -> None:
    """Use one named 1024 source and encode an exact 1536 refinement latent."""

    case = next(case for case in cases() if case.is_refinement)
    built = StrategyComparisonWorkflowBuilder().build(
        case,
        run_id="run-1",
        mask_names=("left.png", "right.png"),
        shared_source_name="source.png",
    )
    load = next(
        node for node in built.prompt.values() if node["class_type"] == "LoadImage"
    )
    scale = next(
        node for node in built.prompt.values() if node["class_type"] == "ImageScale"
    )
    load_inputs = load["inputs"]
    assert isinstance(load_inputs, dict)
    assert load_inputs["image"] == "source.png"
    inputs = scale["inputs"]
    assert isinstance(inputs, dict)
    assert inputs["image"] == next(
        [node_id, 0]
        for node_id, node in built.prompt.items()
        if node["class_type"] == "LoadImage"
    )
    assert inputs["upscale_method"] == "lanczos"
    assert inputs["width"] == 1536
    assert inputs["height"] == 1536
    assert inputs["crop"] == "disabled"
