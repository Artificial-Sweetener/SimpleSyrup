# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the External LLM Prompt Comfy v3 wrapper."""

from __future__ import annotations

from typing import Any

from simple_syrup.nodes.external_llm_prompt import ExternalLLMPrompt
from simple_syrup.nodes_v3.external_llm_prompt import ExternalLLMPromptV3


class FakeExternalLLMService:
    """Service double for v3 wrapper tests."""

    def model_choices(self) -> list[str]:
        """Return deterministic model choices."""

        return ["model-a"]

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        reasoning_effort: str = "default",
        image: object | None = None,
    ) -> str:
        """Return a deterministic response."""

        return (
            f"{model}:{system_prompt}:{user_prompt}:"
            f"{max_tokens}:{reasoning_effort}:{image}"
        )


def test_external_llm_prompt_v3_schema_includes_max_tokens() -> None:
    """The v3 schema exposes the response token limit."""

    schema = ExternalLLMPromptV3.define_schema()

    assert schema.node_id == "SimpleSyrup.ExternalLLMPrompt"
    assert [input_item.id for input_item in schema.inputs] == [
        "model",
        "system_prompt",
        "user_prompt",
        "max_tokens",
        "reasoning_effort",
        "image",
    ]
    max_tokens = schema.inputs[3]
    assert max_tokens.io_type == "INT"
    assert max_tokens.default == 1024
    assert max_tokens.min == 1
    assert max_tokens.max == 32768
    reasoning_effort = schema.inputs[4]
    assert reasoning_effort.io_type == "COMBO"
    assert reasoning_effort.options == ["default", "high", "medium", "low", "off"]
    assert reasoning_effort.default == "default"


def test_external_llm_prompt_v3_execute_forwards_max_tokens(
    monkeypatch: Any,
) -> None:
    """The v3 wrapper forwards max_tokens to the legacy implementation."""

    monkeypatch.setattr(ExternalLLMPrompt, "_service", FakeExternalLLMService())

    result = ExternalLLMPromptV3.execute(
        model="model-a",
        system_prompt="system",
        user_prompt="user",
        max_tokens=128,
        reasoning_effort="off",
        image="image",
    )

    assert result == ("model-a:system:user:128:off:image",)
