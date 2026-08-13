# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify P10.3 public workflow construction."""

import pytest

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.regional_visual_benchmark.matrix import (
    RegionalStrategy,
    SpatialProfile,
    source_positions,
    visual_positions,
)
from tools.regional_visual_benchmark.workflow import (
    VisualBenchmarkWorkflowBuilder,
)


def test_source_workflow_uses_only_global_prompt_at_1024() -> None:
    """Keep shared refinement sources strategy-neutral and independently saved."""

    manifest = load_manifest()
    source = source_positions(manifest)[0]
    built = VisualBenchmarkWorkflowBuilder().build_source(
        manifest, source, run_id="run-1"
    )
    classes = [str(node["class_type"]) for node in built.prompt.values()]

    assert classes.count("CLIPTextEncode") == 2
    assert "SimpleSyrup.EncodePromptBatch" not in classes
    assert "KSampler" in classes
    latent = next(
        node
        for node in built.prompt.values()
        if node["class_type"] == "EmptyCosmosLatentVideo"
    )
    assert latent["inputs"] == {
        "width": 1024,
        "height": 1024,
        "length": 1,
        "batch_size": 1,
    }
    positive = next(
        node
        for node in built.prompt.values()
        if node["class_type"] == "CLIPTextEncode"
        and "worst quality" not in str(node["inputs"])
    )
    positive_inputs = positive["inputs"]
    assert isinstance(positive_inputs, dict)
    case = next(value for value in manifest.cases if value.case_id == source.case_id)
    assert positive_inputs["text"] == ", ".join(
        (case.global_prompt, *case.regional_prompts)
    )


def test_visual_workflows_route_all_strategy_profile_pairs() -> None:
    """Bind every pair to its exact public sampler and corrected settings."""

    manifest = load_manifest()
    case = manifest.cases[0]
    selected = {
        (position.strategy, position.spatial_profile): position
        for position in visual_positions(manifest)
        if position.case_id == case.case_id
        and position.seed == manifest.sampling.seeds[0]
    }
    builder = VisualBenchmarkWorkflowBuilder()
    for (strategy, profile), position in selected.items():
        built = builder.build_visual(
            manifest,
            position,
            run_id="run-1",
            mask_names=("left.png", "right.png"),
            source_name="source.png" if position.is_refinement else None,
        )
        sampler = next(
            node
            for node in built.prompt.values()
            if str(node["class_type"]).startswith("SimpleSyrup.KSampler")
        )
        class_type = str(sampler["class_type"])
        assert ("AttentionCoupling" in class_type) == (
            strategy is RegionalStrategy.ATTENTION_COUPLING
        )
        assert ("Tiled" in class_type or "TiledRegion" in class_type) == (
            profile
            in {
                SpatialProfile.TILED_MULTIDIFFUSION,
                SpatialProfile.TILED_MIXTURE_OF_DIFFUSERS,
            }
        )
        assert ("Contextual" in class_type) == profile.value.startswith("contextual_")
        inputs = sampler["inputs"]
        assert isinstance(inputs, dict)
        assert inputs["steps"] == 30
        assert inputs["cfg"] == 4.0
        assert inputs["denoise"] == (0.3 if position.is_refinement else 1.0)


def test_refinement_workflow_loads_and_scales_exact_shared_source() -> None:
    """Use a named 1024 source only through the approved 1536 refinement path."""

    manifest = load_manifest()
    position = next(
        position for position in visual_positions(manifest) if position.is_refinement
    )
    built = VisualBenchmarkWorkflowBuilder().build_visual(
        manifest,
        position,
        run_id="run-1",
        mask_names=("left.png", "right.png"),
        source_name="source.png",
    )
    load = next(
        node for node in built.prompt.values() if node["class_type"] == "LoadImage"
    )
    scale = next(
        node for node in built.prompt.values() if node["class_type"] == "ImageScale"
    )

    assert load["inputs"] == {"image": "source.png"}
    assert scale["inputs"] == {
        "image": next(
            [node_id, 0]
            for node_id, node in built.prompt.items()
            if node["class_type"] == "LoadImage"
        ),
        "upscale_method": "lanczos",
        "width": 1536,
        "height": 1536,
        "crop": "disabled",
    }


def test_source_presence_must_match_spatial_profile() -> None:
    """Reject accidental from-noise spatial graphs and source-fed full graphs."""

    manifest = load_manifest()
    positions = visual_positions(manifest)
    full = next(position for position in positions if not position.is_refinement)
    refinement = next(position for position in positions if position.is_refinement)
    builder = VisualBenchmarkWorkflowBuilder()

    with pytest.raises(ValueError, match="source publication"):
        builder.build_visual(
            manifest,
            full,
            run_id="run-1",
            mask_names=("left.png", "right.png"),
            source_name="source.png",
        )
    with pytest.raises(ValueError, match="source publication"):
        builder.build_visual(
            manifest,
            refinement,
            run_id="run-1",
            mask_names=("left.png", "right.png"),
            source_name=None,
        )
