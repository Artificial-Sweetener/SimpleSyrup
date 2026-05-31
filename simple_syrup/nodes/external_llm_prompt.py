# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for external LLM prompt generation."""

from __future__ import annotations

from typing import Any

from ..domain.external_llm import (
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
    EXTERNAL_LLM_REASONING_EFFORTS,
)
from ..services.external_llm_prompt_service import ExternalLLMPromptService
from . import tooltips

MAX_EXTERNAL_LLM_MAX_TOKENS = 32768


class ExternalLLMPrompt:
    """Expose an OpenAI-compatible external LLM prompt request as a string node."""

    _service = ExternalLLMPromptService()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_TOOLTIPS = (tooltips.EXTERNAL_LLM_RESPONSE_OUTPUT,)
    FUNCTION = "generate"
    CATEGORY = "SimpleSyrup/Prompting"
    DESCRIPTION = "Sends system and user prompts to a configured external LLM provider."
    SEARCH_ALIASES = ["llm", "openai", "prompt", "preprocess", "text"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare cached external LLM prompt inputs."""

        choices = cls._service.model_choices()
        return {
            "required": {
                "model": (
                    choices,
                    {
                        "default": choices[0],
                        "tooltip": tooltips.EXTERNAL_LLM_MODEL_INPUT,
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": tooltips.EXTERNAL_LLM_SYSTEM_PROMPT_INPUT,
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": tooltips.EXTERNAL_LLM_USER_PROMPT_INPUT,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
                        "min": 1,
                        "max": MAX_EXTERNAL_LLM_MAX_TOKENS,
                        "step": 1,
                        "tooltip": tooltips.EXTERNAL_LLM_MAX_TOKENS_INPUT,
                    },
                ),
                "reasoning_effort": (
                    list(EXTERNAL_LLM_REASONING_EFFORTS),
                    {
                        "default": DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
                        "tooltip": tooltips.EXTERNAL_LLM_REASONING_EFFORT_INPUT,
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": tooltips.EXTERNAL_LLM_IMAGE_INPUT},
                ),
            },
        }

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
        reasoning_effort: str = DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
        image: object | None = None,
    ) -> tuple[str]:
        """Return the external LLM assistant response."""

        return (
            self._service.generate(
                model,
                system_prompt,
                user_prompt,
                max_tokens,
                reasoning_effort,
                image=image,
            ),
        )
