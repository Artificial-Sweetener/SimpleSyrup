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


def test_prompt_control_batch_graph_reports_missing_prompt_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Prompt Control dependency raises an actionable error."""

    def fake_import_module(name: str) -> Any:
        if name == "prompt_control.nodes_lazy":
            raise ModuleNotFoundError(name)
        return import_module(name)

    monkeypatch.setattr(
        "simple_syrup.runtime.prompt_control_batch_graph.import_module",
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


def _install_fake_prompt_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a small Prompt Control lazy-node double for graph tests."""

    prompt_control = ModuleType("prompt_control")
    nodes_lazy = ModuleType("prompt_control.nodes_lazy")

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
            graph_utils = import_module("comfy_execution.graph_utils")
            io = import_module("comfy_api.latest").io
            graph = graph_utils.GraphBuilder()
            node = graph.node(
                "PromptControl.PCLazyTextEncodeAdvanced",
                clip=clip,
                text=text,
            )
            return io.NodeOutput(node.out(0), expand=graph.finalize())

    cast(Any, nodes_lazy).PCLazyTextEncodeAdvanced = FakePCLazyTextEncodeAdvanced
    cast(Any, prompt_control).nodes_lazy = nodes_lazy
    monkeypatch.setitem(sys.modules, "prompt_control", prompt_control)
    monkeypatch.setitem(sys.modules, "prompt_control.nodes_lazy", nodes_lazy)
