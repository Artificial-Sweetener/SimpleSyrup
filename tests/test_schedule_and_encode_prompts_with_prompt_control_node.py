# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Prompt-Control schedule and encode node schema."""

from __future__ import annotations

from typing import Any

from simple_syrup.nodes.schedule_and_encode_prompts_with_prompt_control import (
    ScheduleAndEncodePromptsWithPromptControl as LegacyScheduleAndEncode,
)
from simple_syrup.nodes_v3.schedule_and_encode_prompts_with_prompt_control import (
    ScheduleAndEncodePromptsWithPromptControl,
)


def test_legacy_schedule_and_encode_prompt_control_node_contract() -> None:
    """The legacy node exposes the same workflow-facing contract."""

    inputs = LegacyScheduleAndEncode.INPUT_TYPES()

    assert LegacyScheduleAndEncode.RETURN_TYPES == (
        "MODEL",
        "CONDITIONING,CONDITIONING_BATCH",
        "CONDITIONING,CONDITIONING_BATCH",
    )
    assert LegacyScheduleAndEncode.RETURN_NAMES == (
        "model",
        "positive",
        "negative",
    )
    assert LegacyScheduleAndEncode.FUNCTION == "execute"
    assert LegacyScheduleAndEncode.CATEGORY == "SimpleSyrup/Conditioning"
    assert list(inputs["required"]) == [
        "model",
        "clip",
        "positive_prompt",
        "negative_prompt",
    ]
    assert list(inputs["optional"]) == ["encode_style"]
    assert inputs["required"]["model"][0] == "MODEL"
    assert inputs["required"]["clip"][0] == "CLIP"
    assert inputs["optional"]["encode_style"][0] == "STRING"
    assert inputs["optional"]["encode_style"][1]["forceInput"] is True
    assert inputs["required"]["positive_prompt"][1]["multiline"] is False
    assert inputs["required"]["negative_prompt"][1]["multiline"] is False


def test_legacy_schedule_and_encode_prompt_control_execute_delegates(
    monkeypatch: Any,
) -> None:
    """Legacy node execution delegates to the shared runtime builder."""

    calls: list[dict[str, Any]] = []

    class FakeBuilder:
        """Runtime builder double."""

        def build(self, **kwargs: Any) -> str:
            """Record builder arguments and return a fixed output."""

            calls.append(kwargs)
            return "legacy-node-output"

    monkeypatch.setattr(
        "simple_syrup.nodes.schedule_and_encode_prompts_with_prompt_control."
        "PromptControlScheduleEncodeGraphBuilder",
        FakeBuilder,
    )

    output = LegacyScheduleAndEncode().execute(
        model="model",
        clip="clip",
        encode_style="STYLE(A1111) ",
        positive_prompt="positive",
        negative_prompt="negative",
    )

    assert output == "legacy-node-output"
    assert calls == [
        {
            "model": "model",
            "clip": "clip",
            "encode_style": "STYLE(A1111) ",
            "positive_prompt": "positive",
            "negative_prompt": "negative",
        }
    ]


def test_legacy_schedule_and_encode_prompt_control_omits_encode_style(
    monkeypatch: Any,
) -> None:
    """Legacy node execution no-ops style behavior when the optional socket is empty."""

    calls: list[dict[str, Any]] = []

    class FakeBuilder:
        """Runtime builder double."""

        def build(self, **kwargs: Any) -> str:
            """Record builder arguments and return a fixed output."""

            calls.append(kwargs)
            return "legacy-node-output"

    monkeypatch.setattr(
        "simple_syrup.nodes.schedule_and_encode_prompts_with_prompt_control."
        "PromptControlScheduleEncodeGraphBuilder",
        FakeBuilder,
    )

    output = LegacyScheduleAndEncode().execute(
        model="model",
        clip="clip",
        positive_prompt="positive",
        negative_prompt="negative",
    )

    assert output == "legacy-node-output"
    assert calls == [
        {
            "model": "model",
            "clip": "clip",
            "encode_style": "",
            "positive_prompt": "positive",
            "negative_prompt": "negative",
        }
    ]


def test_schedule_and_encode_prompt_control_node_schema() -> None:
    """The v3 node exposes the planned schedule and encode contract."""

    schema = ScheduleAndEncodePromptsWithPromptControl.define_schema()

    assert schema.node_id == "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl"
    assert schema.display_name == "Schedule & Encode Prompts"
    assert schema.enable_expand is True
    assert schema.category == "SimpleSyrup/Conditioning"
    assert [output.io_type for output in schema.outputs] == [
        "MODEL",
        "CONDITIONING,CONDITIONING_BATCH",
        "CONDITIONING,CONDITIONING_BATCH",
    ]
    assert [output.id for output in schema.outputs] == [
        "model",
        "positive",
        "negative",
    ]
    assert [input_item.id for input_item in schema.inputs] == [
        "model",
        "clip",
        "encode_style",
        "positive_prompt",
        "negative_prompt",
    ]
    assert schema.inputs[2].optional is True


def test_schedule_and_encode_prompt_control_input_types() -> None:
    """The finalized v3 schema exposes Comfy-compatible sockets."""

    inputs = ScheduleAndEncodePromptsWithPromptControl.INPUT_TYPES()

    assert ScheduleAndEncodePromptsWithPromptControl.RETURN_TYPES == [
        "MODEL",
        "CONDITIONING,CONDITIONING_BATCH",
        "CONDITIONING,CONDITIONING_BATCH",
    ]
    assert ScheduleAndEncodePromptsWithPromptControl.RETURN_NAMES == [
        "model",
        "positive",
        "negative",
    ]
    assert list(inputs["required"]) == [
        "model",
        "clip",
        "positive_prompt",
        "negative_prompt",
    ]
    assert list(inputs["optional"]) == ["encode_style"]
    assert inputs["required"]["model"][0] == "MODEL"
    assert inputs["required"]["clip"][0] == "CLIP"
    assert inputs["optional"]["encode_style"][0] == "STRING"
    assert inputs["optional"]["encode_style"][1]["forceInput"] is True
    assert inputs["required"]["positive_prompt"][1]["multiline"] is False
    assert inputs["required"]["negative_prompt"][1]["multiline"] is False


def test_schedule_and_encode_prompt_control_execute_delegates(
    monkeypatch: Any,
) -> None:
    """Node execution delegates behavior to the runtime graph builder."""

    calls: list[dict[str, Any]] = []

    class FakeBuilder:
        """Runtime builder double."""

        def build(self, **kwargs: Any) -> str:
            """Record builder arguments and return a fixed output."""

            calls.append(kwargs)
            return "node-output"

    monkeypatch.setattr(
        "simple_syrup.nodes_v3.schedule_and_encode_prompts_with_prompt_control."
        "PromptControlScheduleEncodeGraphBuilder",
        FakeBuilder,
    )

    output = ScheduleAndEncodePromptsWithPromptControl.execute(
        model="model",
        clip="clip",
        encode_style="STYLE(A1111) ",
        positive_prompt="positive",
        negative_prompt="negative",
    )

    assert output == "node-output"
    assert calls == [
        {
            "model": "model",
            "clip": "clip",
            "encode_style": "STYLE(A1111) ",
            "positive_prompt": "positive",
            "negative_prompt": "negative",
        }
    ]


def test_schedule_and_encode_prompt_control_omits_encode_style(
    monkeypatch: Any,
) -> None:
    """Node execution no-ops style behavior when the optional socket is empty."""

    calls: list[dict[str, Any]] = []

    class FakeBuilder:
        """Runtime builder double."""

        def build(self, **kwargs: Any) -> str:
            """Record builder arguments and return a fixed output."""

            calls.append(kwargs)
            return "node-output"

    monkeypatch.setattr(
        "simple_syrup.nodes_v3.schedule_and_encode_prompts_with_prompt_control."
        "PromptControlScheduleEncodeGraphBuilder",
        FakeBuilder,
    )

    output = ScheduleAndEncodePromptsWithPromptControl.execute(
        model="model",
        clip="clip",
        positive_prompt="positive",
        negative_prompt="negative",
    )

    assert output == "node-output"
    assert calls == [
        {
            "model": "model",
            "clip": "clip",
            "encode_style": "",
            "positive_prompt": "positive",
            "negative_prompt": "negative",
        }
    ]
