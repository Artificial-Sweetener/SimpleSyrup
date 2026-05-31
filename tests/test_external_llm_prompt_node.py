# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the legacy External LLM Prompt node."""

from __future__ import annotations

from simple_syrup.nodes.external_llm_prompt import ExternalLLMPrompt
from simple_syrup.services.external_llm_prompt_service import CONFIGURE_EXTERNAL_LLM


class FakeExternalLLMService:
    """Service double for node tests."""

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


def test_external_llm_prompt_input_types_are_cached_and_single_line() -> None:
    """The node exposes cached model choices and single-line prompt widgets."""

    original_service = ExternalLLMPrompt._service
    ExternalLLMPrompt._service = FakeExternalLLMService()  # type: ignore[assignment]
    try:
        input_types = ExternalLLMPrompt.INPUT_TYPES()
    finally:
        ExternalLLMPrompt._service = original_service

    inputs = input_types["required"]
    assert inputs["model"][0] == ["model-a"]
    assert inputs["model"][1]["default"] == "model-a"
    assert inputs["system_prompt"][0] == "STRING"
    assert inputs["system_prompt"][1]["multiline"] is False
    assert inputs["user_prompt"][0] == "STRING"
    assert inputs["user_prompt"][1]["multiline"] is False
    assert inputs["max_tokens"][0] == "INT"
    assert inputs["max_tokens"][1]["default"] == 1024
    assert inputs["max_tokens"][1]["min"] == 1
    assert inputs["max_tokens"][1]["max"] == 32768
    assert inputs["reasoning_effort"][0] == [
        "default",
        "high",
        "medium",
        "low",
        "off",
    ]
    assert inputs["reasoning_effort"][1]["default"] == "default"
    assert input_types["optional"]["image"][0] == "IMAGE"


def test_external_llm_prompt_output_contract() -> None:
    """The node returns one named STRING output."""

    assert ExternalLLMPrompt.RETURN_TYPES == ("STRING",)
    assert ExternalLLMPrompt.RETURN_NAMES == ("response",)
    assert ExternalLLMPrompt.FUNCTION == "generate"


def test_external_llm_prompt_executes_service() -> None:
    """Node execution delegates to the prompt service."""

    node = ExternalLLMPrompt()
    original_service = node._service
    node._service = FakeExternalLLMService()  # type: ignore[assignment]
    try:
        result = node.generate(
            "model-a",
            "system",
            "user",
            128,
            "off",
            image="image",
        )
    finally:
        node._service = original_service

    assert result == ("model-a:system:user:128:off:image",)


def test_external_llm_prompt_sentinel_constant_matches_plan() -> None:
    """The setup sentinel is stable for dropdown fallback behavior."""

    assert CONFIGURE_EXTERNAL_LLM == "Configure external LLM endpoint"
