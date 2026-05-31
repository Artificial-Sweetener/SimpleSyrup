# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for v3 wrappers that delegate to existing implementation classes."""

from __future__ import annotations

from typing import Any, ClassVar

from simple_syrup.nodes_v3.legacy_node_wrappers import LegacyNodeV3Adapter


class _FakeHidden:
    """Hidden holder double for adapter execution tests."""

    prompt = {"node": "metadata"}


class _FakeLegacyNode:
    """Legacy implementation double with visible and hidden inputs."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    OUTPUT_TOOLTIPS = ("Delegated result.",)
    FUNCTION = "run"
    CATEGORY = "SimpleSyrup/Test"
    DESCRIPTION = "Delegates test inputs."
    SEARCH_ALIASES = ["adapter"]

    calls: ClassVar[list[tuple[str, object]]] = []

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...] | str]]:
        """Return a small legacy contract with one hidden input."""

        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "hello", "tooltip": "Text to delegate."},
                ),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    def run(self, text: str, prompt: object | None = None) -> tuple[str]:
        """Record delegated inputs and return the visible value."""

        self.calls.append((text, prompt))
        return (text,)


class _FakeAdapter(LegacyNodeV3Adapter):
    """Concrete adapter used to verify generic wrapper behavior."""

    LEGACY_NODE_CLASS = _FakeLegacyNode
    NODE_ID = "SimpleSyrup.FakeAdapter"
    DISPLAY_NAME = "Fake Adapter"
    hidden = _FakeHidden()


def test_legacy_node_v3_adapter_builds_schema() -> None:
    """The adapter converts legacy metadata into a v3 schema."""

    schema = _FakeAdapter.define_schema()

    assert schema.node_id == "SimpleSyrup.FakeAdapter"
    assert schema.display_name == "Fake Adapter"
    assert schema.category == "SimpleSyrup/Test"
    assert schema.inputs[0].id == "text"
    assert schema.inputs[0].tooltip == "Text to delegate."
    assert schema.hidden[0].value == "PROMPT"
    assert schema.outputs[0].id == "result"
    assert schema.outputs[0].tooltip == "Delegated result."


def test_legacy_node_v3_adapter_delegates_execution_with_hidden_inputs() -> None:
    """The adapter passes v3 visible and hidden values to the implementation."""

    _FakeLegacyNode.calls.clear()

    assert _FakeAdapter.execute(text="value") == ("value",)
    assert _FakeLegacyNode.calls == [("value", {"node": "metadata"})]
