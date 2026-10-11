# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare capture at Comfy's native validation boundary for every queue caller."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from importlib import import_module
from typing import Any, Protocol, cast

from ..shared.logging import get_logger
from .noise_reuse_graph import rewrite_noise_reuse_graph

LOGGER = get_logger(__name__)
_MARKER = "_simple_syrup_noise_reuse_validation"
NativePromptValidator = Callable[
    [str, dict[str, Any], list[str] | None],
    Awaitable[tuple[bool, object, object, object]],
]


class NativeValidationModule(Protocol):
    """Describe the host's replaceable async prompt validation boundary."""

    validate_prompt: NativePromptValidator


def prepare_noise_reuse_prompt(prompt: dict[str, Any]) -> None:
    """Install cache-keyed capture dependencies before source sampling executes."""
    rewrite_noise_reuse_graph(prompt, import_module("nodes").NODE_CLASS_MAPPINGS)


class NoiseReuseValidationAdapter:
    """Own capture preparation while preserving the host validator's contract."""

    def __init__(self, validator: NativePromptValidator) -> None:
        """Retain the current validator, including other installed extensions."""
        self._validator = validator

    def wrap(self) -> NativePromptValidator:
        """Preserve the host's async function identity and introspection metadata."""

        @wraps(self._validator)
        async def validate_prompt(
            prompt_id: str,
            prompt: dict[str, Any],
            partial_execution_list: list[str] | None,
        ) -> tuple[bool, object, object, object]:
            """Delegate through this adapter without changing the native signature."""
            return await self(prompt_id, prompt, partial_execution_list)

        setattr(validate_prompt, _MARKER, True)
        return validate_prompt

    async def __call__(
        self,
        prompt_id: str,
        prompt: dict[str, Any],
        partial_execution_list: list[str] | None,
    ) -> tuple[bool, object, object, object]:
        """Prepare once before native validation and propagate its exact result."""
        try:
            prepare_noise_reuse_prompt(prompt)
        except Exception:
            LOGGER.exception(
                "Noise reuse capture preparation failed before Comfy validation",
                extra={"prompt_id": prompt_id, "operation": "prepare_capture"},
            )
            raise
        return await self._validator(prompt_id, prompt, partial_execution_list)


def register_noise_reuse_validation() -> None:
    """Install once on the shared boundary used by HTTP and direct graph queues."""
    try:
        execution = cast(NativeValidationModule, import_module("execution"))
    except ModuleNotFoundError as error:
        if error.name != "execution":
            raise
        return
    validator = execution.validate_prompt
    if not getattr(validator, _MARKER, False):
        execution.validate_prompt = NoiseReuseValidationAdapter(validator).wrap()
        LOGGER.info("Installed noise reuse capture preparation before Comfy validation")
