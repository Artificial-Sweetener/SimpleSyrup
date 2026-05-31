# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Domain objects and validation for external LLM provider integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import urlparse

DEFAULT_EXTERNAL_LLM_MAX_TOKENS = 1024
DEFAULT_EXTERNAL_LLM_REASONING_EFFORT = "default"
EXTERNAL_LLM_REASONING_EFFORTS = (
    DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
    "high",
    "medium",
    "low",
    "off",
)


class ExternalLLMConfigError(ValueError):
    """Raised when external LLM configuration is invalid."""


class ExternalLLMProviderError(RuntimeError):
    """Raised when an external LLM provider request fails."""


@dataclass(frozen=True)
class ExternalLLMConfig:
    """Validated non-secret external LLM provider configuration."""

    base_url: str
    cached_models: tuple[str, ...] = ()
    default_model: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""

        normalized_url = normalize_base_url(self.base_url)
        models = normalize_model_ids(self.cached_models)
        default = self.default_model.strip()
        if default and models and default not in models:
            raise ExternalLLMConfigError(
                "Default external LLM model must be one of the cached models."
            )
        object.__setattr__(self, "base_url", normalized_url)
        object.__setattr__(self, "cached_models", models)
        object.__setattr__(self, "default_model", default)


@dataclass(frozen=True)
class ExternalLLMModel:
    """A provider model advertised through an OpenAI-compatible models response."""

    id: str

    def __post_init__(self) -> None:
        """Reject empty provider model identifiers."""

        model_id = self.id.strip()
        if not model_id:
            raise ExternalLLMConfigError("External LLM model id must not be empty.")
        object.__setattr__(self, "id", model_id)


@dataclass(frozen=True)
class ExternalLLMChatRequest:
    """Validated chat completion request fields for an external LLM provider."""

    VALID_REASONING_EFFORTS: ClassVar[tuple[str, ...]] = EXTERNAL_LLM_REASONING_EFFORTS

    model: str
    system_prompt: str
    user_prompt: str
    max_tokens: int = DEFAULT_EXTERNAL_LLM_MAX_TOKENS
    reasoning_effort: str = DEFAULT_EXTERNAL_LLM_REASONING_EFFORT
    image_data_url: str | None = None

    def __post_init__(self) -> None:
        """Validate prompt request values before provider execution."""

        model = self.model.strip()
        reasoning_effort = self.reasoning_effort.strip()
        if not model:
            raise ExternalLLMConfigError("External LLM model must not be empty.")
        if not self.user_prompt.strip():
            raise ExternalLLMConfigError(
                "User prompt must not be empty for external LLM requests."
            )
        if self.max_tokens < 1:
            raise ExternalLLMConfigError("External LLM max tokens must be at least 1.")
        if reasoning_effort not in self.VALID_REASONING_EFFORTS:
            choices = ", ".join(self.VALID_REASONING_EFFORTS)
            raise ExternalLLMConfigError(
                f"External LLM reasoning effort must be one of: {choices}."
            )
        if self.image_data_url is not None and not self.image_data_url.startswith(
            "data:image/"
        ):
            raise ExternalLLMConfigError(
                "External LLM image input must be an image data URL."
            )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "reasoning_effort", reasoning_effort)


@dataclass(frozen=True)
class ExternalLLMChatResponse:
    """Assistant response content returned by an external LLM provider."""

    content: str


def normalize_base_url(base_url: str) -> str:
    """Return a normalized absolute HTTP(S) endpoint URL."""

    candidate = base_url.strip().rstrip("/")
    if not candidate:
        raise ExternalLLMConfigError(
            "Configure an external LLM endpoint in SimpleSyrup settings before "
            "using this node."
        )

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ExternalLLMConfigError(
            "External LLM endpoint must be an absolute http:// or https:// URL."
        )
    return candidate


def normalize_model_ids(values: object) -> tuple[str, ...]:
    """Return non-empty de-duplicated model ids in provider order."""

    if not isinstance(values, (list, tuple)):
        raise ExternalLLMConfigError("External LLM cached models must be a list.")

    models: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ExternalLLMConfigError(
                "External LLM cached model ids must be strings."
            )
        model = value.strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return tuple(models)
