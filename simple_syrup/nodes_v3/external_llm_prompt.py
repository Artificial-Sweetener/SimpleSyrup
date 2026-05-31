# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for external LLM prompt generation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.external_llm import (
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
    EXTERNAL_LLM_REASONING_EFFORTS,
)
from ..nodes import tooltips
from ..nodes.external_llm_prompt import (
    MAX_EXTERNAL_LLM_MAX_TOKENS,
    ExternalLLMPrompt,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class ExternalLLMPromptV3(_ComfyNodeBase):
    """Expose External LLM Prompt through Comfy's v3 extension API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the External LLM Prompt v3 schema."""

        required = ExternalLLMPrompt.INPUT_TYPES()["required"]
        model_choices = list(required["model"][0])
        model_default = str(required["model"][1]["default"])
        return _comfy_io.Schema(
            node_id="SimpleSyrup.ExternalLLMPrompt",
            display_name="External LLM Prompt",
            category="SimpleSyrup/Prompting",
            description=(
                "Sends system and user prompts to a configured external LLM provider."
            ),
            search_aliases=["llm", "openai", "prompt", "preprocess", "text"],
            inputs=[
                _comfy_io.Combo.Input(
                    "model",
                    options=model_choices,
                    default=model_default,
                    tooltip=tooltips.EXTERNAL_LLM_MODEL_INPUT,
                ),
                _comfy_io.String.Input(
                    "system_prompt",
                    multiline=False,
                    default="",
                    tooltip=tooltips.EXTERNAL_LLM_SYSTEM_PROMPT_INPUT,
                ),
                _comfy_io.String.Input(
                    "user_prompt",
                    multiline=False,
                    default="",
                    tooltip=tooltips.EXTERNAL_LLM_USER_PROMPT_INPUT,
                ),
                _comfy_io.Int.Input(
                    "max_tokens",
                    default=DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
                    min=1,
                    max=MAX_EXTERNAL_LLM_MAX_TOKENS,
                    step=1,
                    tooltip=tooltips.EXTERNAL_LLM_MAX_TOKENS_INPUT,
                ),
                _comfy_io.Combo.Input(
                    "reasoning_effort",
                    options=list(EXTERNAL_LLM_REASONING_EFFORTS),
                    default=DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
                    tooltip=tooltips.EXTERNAL_LLM_REASONING_EFFORT_INPUT,
                ),
                _comfy_io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=tooltips.EXTERNAL_LLM_IMAGE_INPUT,
                ),
            ],
            outputs=[
                _comfy_io.String.Output(
                    "response",
                    tooltip=tooltips.EXTERNAL_LLM_RESPONSE_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
        reasoning_effort: str = DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
        image: object | None = None,
    ) -> tuple[str]:
        """Run the legacy node implementation behind the v3 schema."""

        return ExternalLLMPrompt().generate(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            image=image,
        )
