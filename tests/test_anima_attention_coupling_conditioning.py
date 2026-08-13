# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the accepted P5/P6/P7 scheduled Anima conditioning graph."""

from __future__ import annotations

from typing import cast

from tools.anima_attention_coupling_integration.matrix import cases
from tools.anima_attention_coupling_integration.workflow import (
    IntegrationWorkflowBuilder,
)
from tools.comfy_api import JsonObject


def test_conditioning_preserves_global_and_regional_lora_ownership() -> None:
    """Keep global model LoRAs outside exact region-segment prompt tags."""

    workflow = IntegrationWorkflowBuilder().build(
        cases()[2],
        run_id="run",
        mask_names=("left.png", "right.png"),
    )

    global_loader = _sole(workflow.prompt, "LoraLoaderModelOnly")
    assert "GLOBAL_ADAPTER" in str(_inputs(global_loader)["lora_name"])
    encoded = _sole(
        workflow.prompt,
        "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
    )
    prompt = str(_inputs(encoded)["positive_prompt"])
    assert prompt.count("[SEP]") == 2
    assert prompt.index("ADAPTER_A") < prompt.index("ADAPTER_B")
    assert _inputs(encoded)["model"] == [
        next(iter(_node_ids(workflow.prompt, global_loader))),
        0,
    ]


def test_conditioning_renders_exact_adjacent_schedule_intervals() -> None:
    """Pin exact Prompt Control interval syntax at the prompt-policy owner."""

    workflow = IntegrationWorkflowBuilder().build(
        cases()[4],
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    encoded = _sole(
        workflow.prompt,
        "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
    )
    prompt = str(_inputs(encoded)["positive_prompt"])

    assert "[<lora:Anima\\style\\adapter-a.safetensors:0.8>:0.00,0.50]" in prompt
    assert "[<lora:Anima\\style\\adapter-b.safetensors:0.8>:0.50,1.00]" in prompt


def _sole(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return the sole node with one required class."""

    matches = [node for node in prompt.values() if node["class_type"] == class_type]
    assert len(matches) == 1
    return matches[0]


def _node_ids(
    prompt: dict[str, JsonObject],
    expected: JsonObject,
) -> tuple[str, ...]:
    """Return graph IDs whose values are the expected node object."""

    return tuple(node_id for node_id, node in prompt.items() if node is expected)


def _inputs(node: JsonObject) -> dict[str, object]:
    """Narrow one graph node's inputs."""

    return cast(dict[str, object], node["inputs"])
