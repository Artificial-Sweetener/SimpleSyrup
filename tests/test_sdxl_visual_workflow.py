# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify one-case-at-a-time U11 managed SDXL workflow graphs."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_cases import visual_cases
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    build_sdxl_visual_workflow,
)


def test_baseline_builds_only_one_native_full_sampler(tmp_path: Path) -> None:
    """Keep one source trajectory and four authored G/L encodes."""

    case = next(
        item
        for item in visual_cases(visual_inventory(tmp_path))
        if item.case_id == "baseline"
    )
    built = build_sdxl_visual_workflow(
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
    )
    classes = _class_counts(built.prompt)

    assert [output.artifact_id for output in built.outputs] == ["baseline--full"]
    assert classes["SimpleSyrup.KSamplerAttentionCoupling"] == 1
    assert "SimpleSyrup.KSamplerAttentionCouplingTiled" not in classes
    assert "SimpleSyrup.KSamplerAttentionCouplingContextual" not in classes
    assert classes["CLIPTextEncodeSDXL"] == 4
    assert "CLIPTextEncode" not in classes
    assert "CreateHookLora" not in classes
    assert "LoraLoader" not in classes
    sampler = next(
        node["inputs"]
        for node in built.prompt.values()
        if node["class_type"] == "SimpleSyrup.KSamplerAttentionCoupling"
    )
    assert isinstance(sampler, dict)
    assert sampler["regional_prompt_weight"] == 0.4
    assert sampler["region_mask_feather"] == 0


def test_visual_workflow_applies_one_explicit_seed_to_every_sampler(
    tmp_path: Path,
) -> None:
    """Keep every mode and durable workflow identity on one requested seed."""

    case = next(
        item
        for item in visual_cases(visual_inventory(tmp_path))
        if item.case_id == "spatial-mode-global-style-regional-character"
    )
    built = build_sdxl_visual_workflow(
        run_id="seed-control",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
        seed=7_429_113_058,
    )

    assert built.seed == 7_429_113_058
    assert _sampler_seeds(built.prompt) == {7_429_113_058}


def test_full_regional_control_changes_only_sampler_prompt_weight(
    tmp_path: Path,
) -> None:
    """Emit the case-authored full strength through the public sampler input."""

    cases = visual_cases(visual_inventory(tmp_path))
    by_id = {case.case_id: case for case in cases}
    control = build_sdxl_visual_workflow(
        run_id="run-control",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=by_id["global-style-control"],
    )
    candidate = build_sdxl_visual_workflow(
        run_id="run-candidate",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=by_id["global-style-full-regional-control"],
    )

    control_sampler = _sampler_inputs(control.prompt)
    candidate_sampler = _sampler_inputs(candidate.prompt)
    assert control_sampler["regional_prompt_weight"] == 0.4
    assert candidate_sampler["regional_prompt_weight"] == 1.0
    differing_inputs = {
        name
        for name in control_sampler
        if control_sampler[name] != candidate_sampler[name]
    }
    assert differing_inputs == {"regional_prompt_weight"}


def test_global_style_character_pair_emits_full_regional_prompt_weight(
    tmp_path: Path,
) -> None:
    """Keep the prompt control and regional-character candidate at full strength."""

    by_id = {case.case_id: case for case in visual_cases(visual_inventory(tmp_path))}
    for case_id in (
        "global-style-right-character-prompt-control",
        "global-style-regional-character",
    ):
        built = build_sdxl_visual_workflow(
            run_id=f"run-{case_id}",
            checkpoint_name=r"owned\checkpoint.safetensors",
            mask_names=("left.png", "right.png"),
            case=by_id[case_id],
        )
        assert _sampler_inputs(built.prompt)["regional_prompt_weight"] == 1.0


def test_selected_case_builds_full_tiled_and_contextual_from_one_source(
    tmp_path: Path,
) -> None:
    """Refine only the accepted global-style plus regional-character source."""

    case = next(
        item
        for item in visual_cases(visual_inventory(tmp_path))
        if item.case_id == "spatial-mode-global-style-regional-character"
    )
    built = build_sdxl_visual_workflow(
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
    )
    classes = _class_counts(built.prompt)

    assert [output.artifact_id for output in built.outputs] == [
        "spatial-mode-global-style-regional-character--full",
        "spatial-mode-global-style-regional-character--tiled-1.5x",
        "spatial-mode-global-style-regional-character--contextual-1.5x",
    ]
    assert classes["SimpleSyrup.KSamplerAttentionCoupling"] == 1
    assert classes["SimpleSyrup.KSamplerAttentionCouplingTiled"] == 1
    assert classes["SimpleSyrup.KSamplerAttentionCouplingContextual"] == 1
    assert classes["ImageScale"] == 1
    assert classes["VAEEncode"] == 1
    assert classes["VAEDecode"] == 3
    assert classes["LoraLoader"] == 1
    assert classes["CreateHookLora"] == 1


def _class_counts(prompt: dict[str, JsonObject]) -> dict[str, int]:
    """Count exact API node classes for concise graph assertions."""

    counts: dict[str, int] = {}
    for node in prompt.values():
        class_type = node["class_type"]
        if not isinstance(class_type, str):
            raise AssertionError("Generated class_type must be text.")
        counts[class_type] = counts.get(class_type, 0) + 1
    return counts


def _sampler_inputs(prompt: dict[str, JsonObject]) -> JsonObject:
    """Return the sole full attention-coupling sampler inputs."""

    samplers = [
        node["inputs"]
        for node in prompt.values()
        if node["class_type"] == "SimpleSyrup.KSamplerAttentionCoupling"
    ]
    if len(samplers) != 1 or not isinstance(samplers[0], dict):
        raise AssertionError("Expected one full attention-coupling sampler.")
    return samplers[0]


def _sampler_seeds(prompt: dict[str, JsonObject]) -> set[int]:
    """Return every explicitly emitted SimpleSyrup sampler seed."""

    seeds: set[int] = set()
    for node in prompt.values():
        class_type = node["class_type"]
        if not isinstance(class_type, str) or not class_type.startswith(
            "SimpleSyrup.KSampler"
        ):
            continue
        inputs = node["inputs"]
        if not isinstance(inputs, dict):
            raise AssertionError("Sampler inputs must be a JSON object.")
        seed = inputs.get("seed")
        if not isinstance(seed, int):
            raise AssertionError("Sampler seed must be an integer.")
        seeds.add(seed)
    return seeds
