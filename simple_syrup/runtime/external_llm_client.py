# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OpenAI-compatible HTTP client for external LLM providers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol, cast

from ..domain.external_llm import (
    ExternalLLMChatRequest,
    ExternalLLMChatResponse,
    ExternalLLMProviderError,
    normalize_base_url,
)
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
MODELS_TIMEOUT_SECONDS = 10
CHAT_TIMEOUT_SECONDS = 60


class UrlopenResponse(Protocol):
    """Subset of urllib response behavior used by the client."""

    status: int

    def read(self) -> bytes:
        """Read response bytes."""

    def __enter__(self) -> UrlopenResponse:
        """Enter a response context manager."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit a response context manager."""


Urlopen = Callable[[urllib.request.Request, int], UrlopenResponse]


class ExternalLLMClient:
    """Call OpenAI-compatible model listing and chat completion endpoints."""

    def __init__(self, urlopen: Urlopen | None = None) -> None:
        """Create a client with injectable HTTP transport."""

        self._urlopen = urlopen or _default_urlopen

    def list_models(self, base_url: str, api_key: str) -> tuple[str, ...]:
        """Return provider model ids from `/models`."""

        payload = self._request_json(
            base_url=base_url,
            path="/models",
            api_key=api_key,
            method="GET",
            body=None,
            timeout=MODELS_TIMEOUT_SECONDS,
        )
        return parse_models_payload(payload)

    def create_chat_completion(
        self,
        base_url: str,
        api_key: str,
        request: ExternalLLMChatRequest,
    ) -> ExternalLLMChatResponse:
        """Return assistant content from `/chat/completions`."""

        messages: list[dict[str, object]] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": _user_message_content(request)},
        ]
        payload = self._request_json(
            base_url=base_url,
            path="/chat/completions",
            api_key=api_key,
            method="POST",
            body=_chat_completion_body(request, messages),
            timeout=CHAT_TIMEOUT_SECONDS,
        )
        return ExternalLLMChatResponse(parse_chat_content(payload))

    def _request_json(
        self,
        base_url: str,
        path: str,
        api_key: str,
        method: str,
        body: dict[str, object] | None,
        timeout: int,
    ) -> object:
        """Send an authenticated provider request and decode JSON."""

        endpoint = f"{normalize_base_url(base_url)}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with self._urlopen(request, timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            LOGGER.warning(
                "external llm provider returned http error",
                extra={"operation": path, "status": error.code},
            )
            raise ExternalLLMProviderError(
                f"External LLM provider returned HTTP {error.code}."
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            LOGGER.warning(
                "external llm provider request failed",
                extra={"operation": path, "reason": str(error)},
            )
            raise ExternalLLMProviderError(
                "External LLM provider request failed before a response was received."
            ) from error

        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ExternalLLMProviderError(
                "External LLM provider returned invalid JSON."
            ) from error


def parse_models_payload(payload: object) -> tuple[str, ...]:
    """Extract de-duplicated model ids from an OpenAI-compatible payload."""

    if not isinstance(payload, dict):
        raise ExternalLLMProviderError(
            "External LLM provider returned an invalid models response."
        )
    data = payload.get("data")
    if not isinstance(data, list):
        raise ExternalLLMProviderError(
            "External LLM provider returned an invalid models response."
        )

    models: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id: Any = item.get("id")
        if not isinstance(model_id, str):
            continue
        model = model_id.strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return tuple(models)


def _default_urlopen(
    request: urllib.request.Request,
    timeout: int,
) -> UrlopenResponse:
    """Open a URL request with an explicit timeout."""

    return cast(UrlopenResponse, urllib.request.urlopen(request, timeout=timeout))


def _user_message_content(request: ExternalLLMChatRequest) -> object:
    """Return text-only or multimodal user message content."""

    if request.image_data_url is None:
        return request.user_prompt
    return [
        {"type": "text", "text": request.user_prompt},
        {
            "type": "image_url",
            "image_url": {"url": request.image_data_url},
        },
    ]


def _chat_completion_body(
    request: ExternalLLMChatRequest,
    messages: list[dict[str, object]],
) -> dict[str, object]:
    """Return an OpenAI-compatible chat request body with optional reasoning."""

    body: dict[str, object] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
    }
    if request.reasoning_effort in {"high", "medium", "low"}:
        body["reasoning_effort"] = request.reasoning_effort
    if request.reasoning_effort == "off":
        body["chat_template_kwargs"] = {"thinking": False}
    return body


def parse_chat_content(payload: object) -> str:
    """Extract assistant message content from a chat completion payload."""

    if not isinstance(payload, dict):
        raise ExternalLLMProviderError(
            "The external LLM provider returned an invalid chat completion response."
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ExternalLLMProviderError(
            "The external LLM provider returned an invalid chat completion response."
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise ExternalLLMProviderError(
            "The external LLM provider returned an invalid chat completion response."
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise ExternalLLMProviderError(
            "The external LLM provider returned an invalid chat completion response."
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise ExternalLLMProviderError(
            "The external LLM provider returned an invalid chat completion response."
        )
    return content
