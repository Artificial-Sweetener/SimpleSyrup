# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the exact-prompt SDXL LoRA baseline case family."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from sdxl_visual_test_inventory import visual_inventory

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.graph import SdxlWorkflowGraph
from tools.sdxl_attention_coupling_integration.visual_conditioning import (
    SdxlVisualConditioningBuilder,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
    lora_baseline_cases,
)


def test_baseline_cases_swap_only_the_active_character_sections(
    tmp_path: Path,
) -> None:
    """Keep inactive regions on their exact no-LoRA section controls."""

    inventory = visual_inventory(tmp_path)
    prompts = _prompts()
    cases = lora_baseline_cases(inventory, prompts)

    assert [case.case_id for case in cases] == [
        "left-character-baseline",
        "right-character-baseline",
        "simultaneous-characters-baseline",
        "global-style-baseline",
    ]
    assert cases[0].left_l == inventory.left_character.prompt_l
    assert cases[0].right_l == prompts.right_positive_l
    assert cases[1].left_l == prompts.left_positive_l
    assert cases[1].right_l == inventory.right_character.prompt_l
    assert cases[2].left_l == inventory.left_character.prompt_l
    assert cases[2].right_l == inventory.right_character.prompt_l
    assert cases[3].left_l == prompts.left_positive_l
    assert cases[3].right_l == prompts.right_positive_l
    assert cases[3].base_positive_g.startswith(
        f"{inventory.style.prompt_g}, {prompts.base_positive_g}"
    )
    assert cases[3].base_positive_l.startswith(
        f"{inventory.style.prompt_l}, {prompts.base_positive_l}"
    )


def test_regional_character_hooks_cover_authored_positive_and_negative_branches(
    tmp_path: Path,
) -> None:
    """Apply each regional adapter consistently across its CFG conditioning pair."""

    inventory = visual_inventory(tmp_path)
    prompts = _prompts()
    cases = lora_baseline_cases(inventory, prompts)

    left_graph = _build(cases[0])
    assert len(_nodes(left_graph, "CreateHookLora")) == 1
    assert len(_nodes(left_graph, "ConditioningSetProperties")) == 2
    assert _encoded_local_prompts(left_graph) == [
        prompts.base_positive_l,
        prompts.base_negative_l,
        f"{prompts.base_positive_l}, {inventory.left_character.prompt_l}",
        f"{prompts.base_negative_l}, {prompts.left_negative_l}",
        f"{prompts.base_positive_l}, {prompts.right_positive_l}",
        f"{prompts.base_negative_l}, {prompts.right_negative_l}",
    ]

    simultaneous_graph = _build(cases[2])
    assert len(_nodes(simultaneous_graph, "CreateHookLora")) == 2
    assert len(_nodes(simultaneous_graph, "ConditioningSetProperties")) == 4


def test_simultaneous_case_keeps_base_exact_and_composes_each_lora_branch(
    tmp_path: Path,
) -> None:
    """Apply shared globals under each LoRA without cross-character prompt reach."""

    inventory = visual_inventory(tmp_path)
    prompts = _prompts()
    case = lora_baseline_cases(inventory, prompts)[2]

    assert _encoded_prompts(_build(case)) == [
        (prompts.base_positive_g, prompts.base_positive_l),
        (prompts.base_negative_g, prompts.base_negative_l),
        (
            f"{prompts.base_positive_g}, {inventory.left_character.prompt_g}",
            f"{prompts.base_positive_l}, {inventory.left_character.prompt_l}",
        ),
        (
            f"{prompts.base_negative_g}, {prompts.left_negative_g}",
            f"{prompts.base_negative_l}, {prompts.left_negative_l}",
        ),
        (
            f"{prompts.base_positive_g}, {inventory.right_character.prompt_g}",
            f"{prompts.base_positive_l}, {inventory.right_character.prompt_l}",
        ),
        (
            f"{prompts.base_negative_g}, {prompts.right_negative_g}",
            f"{prompts.base_negative_l}, {prompts.right_negative_l}",
        ),
    ]


def test_global_style_uses_one_ordinary_loader_and_no_regional_hooks(
    tmp_path: Path,
) -> None:
    """Keep the style adapter global while retaining both section controls."""

    case = lora_baseline_cases(visual_inventory(tmp_path), _prompts())[3]
    graph = _build(case)

    assert len(_nodes(graph, "LoraLoader")) == 1
    assert _nodes(graph, "CreateHookLora") == []
    assert _nodes(graph, "ConditioningSetProperties") == []


def _prompts() -> SdxlVisualPromptSet:
    """Return one identity-neutral complete SEP prompt fixture."""

    return SdxlVisualPromptSet(
        base_positive_g="global positive g",
        base_positive_l="global positive l",
        base_negative_g="global negative g",
        base_negative_l="global negative l",
        left_positive_g="left control g",
        left_positive_l="left control l",
        right_positive_g="right control g",
        right_positive_l="right control l",
        left_negative_g="left negative g",
        left_negative_l="left negative l",
        right_negative_g="right negative g",
        right_negative_l="right negative l",
    )


def _build(case: object) -> dict[str, JsonObject]:
    """Build one declared case against stable graph references."""

    from tools.sdxl_attention_coupling_integration.visual_case_model import (
        SdxlVisualCase,
    )

    if not isinstance(case, SdxlVisualCase):
        raise TypeError("Expected one SDXL visual case.")
    graph = SdxlWorkflowGraph()
    SdxlVisualConditioningBuilder().build(
        graph,
        case=case,
        model=["model", 0],
        clip=["clip", 0],
    )
    return graph.prompt


def _nodes(prompt: dict[str, JsonObject], class_type: str) -> list[JsonObject]:
    """Return graph nodes with one exact type."""

    return [node for node in prompt.values() if node["class_type"] == class_type]


def _encoded_local_prompts(prompt: dict[str, JsonObject]) -> list[str]:
    """Return SDXL local-encoder prompt text in graph order."""

    return [
        cast(str, cast(JsonObject, node["inputs"])["text_l"])
        for node in _nodes(prompt, "CLIPTextEncodeSDXL")
    ]


def _encoded_prompts(prompt: dict[str, JsonObject]) -> list[tuple[str, str]]:
    """Return ordered SDXL global/local prompt pairs."""

    return [
        (
            cast(str, cast(JsonObject, node["inputs"])["text_g"]),
            cast(str, cast(JsonObject, node["inputs"])["text_l"]),
        )
        for node in _nodes(prompt, "CLIPTextEncodeSDXL")
    ]
