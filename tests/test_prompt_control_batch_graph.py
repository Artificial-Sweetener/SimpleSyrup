# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Prompt Control prompt batch lazy graph expansion."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any, cast

import pytest

from simple_syrup.runtime.prompt_control_batch_graph import (
    PROMPT_CONTROL_MISSING_MESSAGE,
    PromptControlBatchGraphBuilder,
)


def test_prompt_control_batch_graph_matches_single_lazy_text_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One positive chunk preserves Prompt Control scheduled prompt expansion."""

    _install_fake_prompt_control(monkeypatch)
    graph_utils = import_module("comfy_execution.graph_utils")
    lazy_nodes = import_module("prompt_control.nodes_lazy")
    graph_utils.GraphBuilder.set_default_prefix("UID", 0, 0)
    expected = lazy_nodes.PCLazyTextEncodeAdvanced.execute(
        clip=[0, 0],
        text="[cat:dog:0.5]",
        tags="",
        start=0.0,
        end=1.0,
        num_steps=0,
    )

    graph_utils.GraphBuilder.set_default_prefix("UID", 0, 0)
    output = PromptControlBatchGraphBuilder().build(
        clip=[0, 0],
        positive_prompt="[cat:dog:0.5]",
        negative_prompt="",
        separator="[SEP]",
    )

    assert output.args[0] == ["UID.0.1.1", 0]
    assert output.args[1] == ["UID.0.3.1", 0]
    assert output.expand is not None
    for node_id, node in expected.expand.items():
        assert output.expand[node_id] == node


def test_prompt_control_batch_graph_builds_pack_chain_for_multiple_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple prompt chunks become one start node plus append nodes."""

    _install_fake_prompt_control(monkeypatch)
    graph_utils = import_module("comfy_execution.graph_utils")
    graph_utils.GraphBuilder.set_default_prefix("BATCH", 0, 0)

    output = PromptControlBatchGraphBuilder().build(
        clip=[0, 0],
        positive_prompt="face [SEP] hair",
        negative_prompt="blur [SEP] noise",
        separator="[SEP]",
    )

    assert output.expand is not None
    node_ids = list(output.expand)
    assert len(node_ids) == len(set(node_ids))
    pack_nodes = [
        node
        for node in output.expand.values()
        if node["class_type"].startswith("SimpleSyrup.ConditioningBatch")
    ]
    assert [node["class_type"] for node in pack_nodes] == [
        "SimpleSyrup.ConditioningBatchStart",
        "SimpleSyrup.ConditioningBatchAppend",
        "SimpleSyrup.ConditioningBatchStart",
        "SimpleSyrup.ConditioningBatchAppend",
    ]
    assert output.args[0] == ["BATCH.0.2.2", 0]
    assert output.args[1] == ["BATCH.0.5.2", 0]


def test_prompt_control_batch_graph_attaches_segment_local_lora_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clip-only encoder shares each aligned hook plan across both sides."""

    calls = _install_fake_prompt_control(monkeypatch)
    graph_utils = import_module("comfy_execution.graph_utils")
    graph_utils.GraphBuilder.set_default_prefix("HOOKS", 0, 0)

    output = PromptControlBatchGraphBuilder().build(
        clip=[0, 0],
        positive_prompt="face <lora:a:1> [SEP] hair <lora:b:1>",
        negative_prompt="blur <lora:c:1> [SEP] noise",
        separator="[SEP]",
    )

    assert output.expand is not None
    parsed_hook_nodes = [
        node
        for node in output.expand.values()
        if node["class_type"] == "PCLoraHooksFromText"
    ]
    assert [node["inputs"]["text"] for node in parsed_hook_nodes] == [
        "<lora:a:1>\n<lora:c:1>",
        "<lora:b:1>",
    ]
    regional_hook_nodes = [
        node
        for node in output.expand.values()
        if node["class_type"] == "SimpleSyrup.PrepareRegionalLoraHooks"
    ]
    assert len(regional_hook_nodes) == 2
    assert not any(
        node["class_type"] == "SetClipHooks" for node in output.expand.values()
    )
    attachment_nodes = [
        node
        for node in output.expand.values()
        if node["class_type"] == "ConditioningSetProperties"
    ]
    assert len(attachment_nodes) == 6
    companion_nodes = [
        node
        for node in output.expand.values()
        if node["class_type"] == "SimpleSyrup.AttachRegionalGlobalConditioning"
    ]
    assert len(companion_nodes) == 2
    assert [call["text"] for call in calls] == [
        "face ",
        "hair ",
        "face ",
        "blur ",
        "noise",
        "blur ",
    ]
    assert calls[0]["clip"] == calls[3]["clip"]
    assert calls[1]["clip"] == calls[4]["clip"]
    assert calls[0]["clip"] != calls[1]["clip"]


def test_prompt_control_batch_graph_matches_missing_negative_with_region_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sole global negative is re-encoded under each regional hook plan."""

    calls = _install_fake_prompt_control(monkeypatch)
    graph_utils = import_module("comfy_execution.graph_utils")
    graph_utils.GraphBuilder.set_default_prefix("REGRESSION", 0, 0)

    PromptControlBatchGraphBuilder().build(
        clip=[0, 0],
        positive_prompt=("global [SEP] left <lora:regional:1> [SEP] right"),
        negative_prompt="bad quality",
        separator="[SEP]",
    )

    assert [call["text"] for call in calls] == [
        "global",
        "left ",
        "right",
        "global",
        "bad quality",
        "bad quality",
        "bad quality",
        "bad quality",
    ]
    assert calls[1]["clip"] == calls[5]["clip"]
    assert calls[3]["clip"] == calls[7]["clip"]
    assert calls[0]["clip"] == calls[4]["clip"]
    assert calls[2]["clip"] == calls[6]["clip"]


def test_prompt_control_batch_graph_reports_missing_prompt_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Prompt Control dependency raises an actionable error."""

    def fake_import_module(name: str) -> Any:
        if name == "prompt_control.nodes_lazy":
            raise ModuleNotFoundError(name)
        return import_module(name)

    monkeypatch.setattr(
        "simple_syrup.runtime.prompt_control_graph_adapter.import_module",
        fake_import_module,
    )

    with pytest.raises(RuntimeError, match="requires comfyui-prompt-control"):
        PromptControlBatchGraphBuilder().build(
            clip=[0, 0],
            positive_prompt="face",
            negative_prompt="",
            separator="[SEP]",
        )
    assert PROMPT_CONTROL_MISSING_MESSAGE.startswith("Encode Prompt Batch")


def _install_fake_prompt_control(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Install a small Prompt Control lazy-node double for graph tests."""

    prompt_control = ModuleType("prompt_control")
    nodes_lazy = ModuleType("prompt_control.nodes_lazy")
    calls: list[dict[str, Any]] = []

    class FakePCLazyLoraLoaderAdvanced:
        """Expand segment LoRAs through Comfy's native hook nodes."""

        @staticmethod
        def execute(
            model: Any,
            clip: Any,
            text: str,
            apply_hooks: bool,
            tags: str,
            start: float,
            end: float,
            num_steps: int,
        ) -> Any:
            """Return a hooked CLIP link built from native graph components."""

            assert model is None
            assert apply_hooks is True
            assert tags == ""
            assert start == 0.0
            assert end == 1.0
            assert num_steps == 0
            graph_utils = import_module("comfy_execution.graph_utils")
            io = import_module("comfy_api.latest").io
            graph = graph_utils.GraphBuilder()
            hooks = graph.node(
                "CreateHookLora",
                lora_name=text,
                strength_model=1.0,
                strength_clip=1.0,
            )
            hooked_clip = graph.node(
                "SetClipHooks",
                clip=clip,
                hooks=hooks.out(0),
                apply_to_conds=True,
                schedule_clip=True,
            )
            return io.NodeOutput(
                None,
                hooked_clip.out(0),
                hooks.out(0),
                expand=graph.finalize(),
            )

    class FakePCLazyTextEncodeAdvanced:
        """Graph-expanding stand-in for Prompt Control's lazy text encoder."""

        @staticmethod
        def execute(
            clip: Any,
            text: str,
            tags: str,
            start: float,
            end: float,
            num_steps: int,
        ) -> Any:
            """Return one lazy text encode node output."""

            del tags, start, end, num_steps
            calls.append({"clip": clip, "text": text})
            graph_utils = import_module("comfy_execution.graph_utils")
            io = import_module("comfy_api.latest").io
            graph = graph_utils.GraphBuilder()
            node = graph.node(
                "PromptControl.PCLazyTextEncodeAdvanced",
                clip=clip,
                text=text,
            )
            return io.NodeOutput(node.out(0), expand=graph.finalize())

    cast(Any, nodes_lazy).PCLazyLoraLoaderAdvanced = FakePCLazyLoraLoaderAdvanced
    cast(Any, nodes_lazy).PCLazyTextEncodeAdvanced = FakePCLazyTextEncodeAdvanced
    cast(Any, prompt_control).nodes_lazy = nodes_lazy
    monkeypatch.setitem(sys.modules, "prompt_control", prompt_control)
    monkeypatch.setitem(sys.modules, "prompt_control.nodes_lazy", nodes_lazy)
    return calls
