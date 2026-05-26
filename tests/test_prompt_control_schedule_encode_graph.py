# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Prompt-Control schedule and encode lazy graph expansion."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any, cast

import pytest

from simple_syrup.runtime.prompt_control_schedule_encode_graph import (
    PROMPT_CONTROL_MISSING_MESSAGE,
    PromptControlScheduleEncodeGraphBuilder,
)


def test_schedule_encode_graph_builds_single_conditioning_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single prompts return direct Prompt-Control conditioning links."""

    calls = _install_fake_prompt_control(monkeypatch)

    output = PromptControlScheduleEncodeGraphBuilder().build(
        model=["model", 0],
        clip=["clip", 0],
        positive_prompt="face <lora:positive:1.0>",
        negative_prompt="blur <lora:negative:0.5>",
        encode_style="STYLE(A1111) ",
    )

    assert output.args[0] == ["lora_negative", 0]
    assert output.args[1] == ["encode_0", 0]
    assert output.args[2] == ["encode_1", 0]
    assert output.expand is not None
    assert not any(
        node["class_type"].startswith("SimpleSyrup.ConditioningBatch")
        for node in output.expand.values()
    )
    assert calls["lora"] == [
        {
            "model": ["model", 0],
            "clip": ["clip", 0],
            "text": "<lora:positive:1.0>",
        },
        {
            "model": ["lora_positive", 0],
            "clip": ["lora_positive", 1],
            "text": "<lora:negative:0.5>",
        },
    ]
    assert calls["encode"] == [
        {"clip": ["lora_negative", 1], "text": "STYLE(A1111) face "},
        {"clip": ["lora_negative", 1], "text": "STYLE(A1111) blur "},
    ]


def test_schedule_encode_graph_packs_only_multichunk_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A side with multiple chunks becomes a conditioning batch."""

    _install_fake_prompt_control(monkeypatch)

    output = PromptControlScheduleEncodeGraphBuilder().build(
        model=["model", 0],
        clip=["clip", 0],
        positive_prompt="face [SEP] hair",
        negative_prompt="blur",
    )

    assert output.args[1] != ["encode_0", 0]
    assert output.args[2] == ["encode_2", 0]
    assert output.expand is not None
    pack_nodes = [
        node
        for node in output.expand.values()
        if node["class_type"].startswith("SimpleSyrup.ConditioningBatch")
    ]
    assert [node["class_type"] for node in pack_nodes] == [
        "SimpleSyrup.ConditioningBatchStart",
        "SimpleSyrup.ConditioningBatchAppend",
    ]


def test_schedule_encode_graph_collects_loras_from_all_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LoRA scheduling sees all tags from every separator-delimited chunk."""

    calls = _install_fake_prompt_control(monkeypatch)

    PromptControlScheduleEncodeGraphBuilder().build(
        model=["model", 0],
        clip=["clip", 0],
        positive_prompt="face <lora:a:1> [SEP] hair <lora:b:1>",
        negative_prompt="blur <lora:c:1> [SEP] noise <lora:d:1>",
    )

    assert calls["lora"][0]["text"] == "<lora:a:1>\n<lora:b:1>"
    assert calls["lora"][1]["text"] == "<lora:c:1>\n<lora:d:1>"


def test_schedule_encode_graph_reports_duplicate_expand_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate generated node ids are rejected with context."""

    _install_fake_prompt_control(monkeypatch, duplicate_lora_ids=True)

    with pytest.raises(ValueError, match="negative LoRA scheduling"):
        PromptControlScheduleEncodeGraphBuilder().build(
            model=["model", 0],
            clip=["clip", 0],
            positive_prompt="face",
            negative_prompt="blur",
        )


def test_schedule_encode_graph_reports_missing_prompt_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Prompt-Control dependency raises an actionable error."""

    def fake_import_module(name: str) -> Any:
        if name == "prompt_control.nodes_lazy":
            raise ModuleNotFoundError(name)
        return import_module(name)

    monkeypatch.setattr(
        "simple_syrup.runtime.prompt_control_schedule_encode_graph.import_module",
        fake_import_module,
    )

    with pytest.raises(RuntimeError, match="requires comfyui-prompt-control"):
        PromptControlScheduleEncodeGraphBuilder().build(
            model=["model", 0],
            clip=["clip", 0],
            positive_prompt="face",
            negative_prompt="blur",
        )
    assert PROMPT_CONTROL_MISSING_MESSAGE.startswith("Schedule & Encode Prompts")


def _install_fake_prompt_control(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duplicate_lora_ids: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Install graph-expanding Prompt-Control stand-ins for builder tests."""

    prompt_control = ModuleType("prompt_control")
    nodes_lazy = ModuleType("prompt_control.nodes_lazy")
    io = import_module("comfy_api.latest").io
    calls: dict[str, list[dict[str, Any]]] = {"lora": [], "encode": []}

    class FakePCLazyLoraLoaderAdvanced:
        """Prompt-Control LoRA scheduler test double."""

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
            """Return deterministic model and clip links."""

            assert apply_hooks is True
            assert tags == ""
            assert start == 0.0
            assert end == 1.0
            assert num_steps == 0
            side = "positive" if not calls["lora"] else "negative"
            calls["lora"].append({"model": model, "clip": clip, "text": text})
            node_id = "duplicate_lora" if duplicate_lora_ids else f"lora_{side}"
            return io.NodeOutput(
                [f"lora_{side}", 0],
                [f"lora_{side}", 1],
                None,
                expand={node_id: {"class_type": "PromptControl.FakeLora"}},
            )

    class FakePCLazyTextEncodeAdvanced:
        """Prompt-Control text encoder test double."""

        @staticmethod
        def execute(
            clip: Any,
            text: str,
            tags: str,
            start: float,
            end: float,
            num_steps: int,
        ) -> Any:
            """Return deterministic conditioning links."""

            assert tags == ""
            assert start == 0.0
            assert end == 1.0
            assert num_steps == 0
            index = len(calls["encode"])
            calls["encode"].append({"clip": clip, "text": text})
            node_id = f"encode_{index}"
            return io.NodeOutput(
                [node_id, 0],
                expand={node_id: {"class_type": "PromptControl.FakeTextEncode"}},
            )

    cast(Any, nodes_lazy).PCLazyLoraLoaderAdvanced = FakePCLazyLoraLoaderAdvanced
    cast(Any, nodes_lazy).PCLazyTextEncodeAdvanced = FakePCLazyTextEncodeAdvanced
    cast(Any, prompt_control).nodes_lazy = nodes_lazy
    monkeypatch.setitem(sys.modules, "prompt_control", prompt_control)
    monkeypatch.setitem(sys.modules, "prompt_control.nodes_lazy", nodes_lazy)
    return calls
