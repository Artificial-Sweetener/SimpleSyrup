# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the OpenAI-compatible external LLM HTTP client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from simple_syrup.domain.external_llm import (
    ExternalLLMChatRequest,
    ExternalLLMProviderError,
)
from simple_syrup.runtime.external_llm_client import ExternalLLMClient


class FakeResponse:
    """Context-managed HTTP response double."""

    status = 200

    def __init__(self, payload: object) -> None:
        """Store JSON payload bytes."""

        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        """Return response bytes."""

        return self._payload

    def __enter__(self) -> FakeResponse:
        """Enter context manager."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit context manager."""


class CapturingUrlopen:
    """Transport double that captures requests."""

    def __init__(self, payload: object) -> None:
        """Store the response payload."""

        self.payload = payload
        self.requests: list[tuple[urllib.request.Request, int]] = []

    def __call__(
        self,
        request: urllib.request.Request,
        timeout: int,
    ) -> FakeResponse:
        """Capture request and return configured response."""

        self.requests.append((request, timeout))
        return FakeResponse(self.payload)


def test_list_models_sends_authenticated_models_request() -> None:
    """Model refresh uses the OpenAI-compatible `/models` endpoint."""

    urlopen = CapturingUrlopen(
        {"data": [{"id": "model-a"}, {"id": "model-a"}, {"id": "model-b"}]}
    )
    client = ExternalLLMClient(urlopen)

    models = client.list_models("https://provider.example/v1/", "secret")

    request, timeout = urlopen.requests[0]
    assert request.full_url == "https://provider.example/v1/models"
    assert request.get_method() == "GET"
    assert request.headers["Authorization"] == "Bearer secret"
    assert timeout == 10
    assert models == ("model-a", "model-b")


def test_list_models_rejects_malformed_response() -> None:
    """Malformed provider model payloads fail clearly."""

    client = ExternalLLMClient(CapturingUrlopen({"data": "bad"}))

    with pytest.raises(ExternalLLMProviderError, match="models response"):
        client.list_models("https://provider.example/v1", "secret")


def test_chat_completion_sends_messages_and_returns_content() -> None:
    """Chat completion uses system and user messages and extracts content."""

    urlopen = CapturingUrlopen(
        {"choices": [{"message": {"content": "rewritten prompt"}}]}
    )
    client = ExternalLLMClient(urlopen)

    response = client.create_chat_completion(
        "https://provider.example/v1",
        "secret",
        ExternalLLMChatRequest(
            model="model-a",
            system_prompt="system",
            user_prompt="user",
        ),
    )

    request, timeout = urlopen.requests[0]
    assert request.full_url == "https://provider.example/v1/chat/completions"
    assert request.get_method() == "POST"
    assert timeout == 60
    assert isinstance(request.data, bytes)
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "model-a",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "max_tokens": 1024,
    }
    assert response.content == "rewritten prompt"


def test_chat_completion_sends_configured_max_tokens() -> None:
    """Chat completion forwards the workflow response token limit."""

    urlopen = CapturingUrlopen({"choices": [{"message": {"content": "ok"}}]})
    client = ExternalLLMClient(urlopen)

    client.create_chat_completion(
        "https://provider.example/v1",
        "secret",
        ExternalLLMChatRequest(
            model="model-a",
            system_prompt="system",
            user_prompt="user",
            max_tokens=128,
        ),
    )

    request, _timeout = urlopen.requests[0]
    assert isinstance(request.data, bytes)
    body = json.loads(request.data.decode("utf-8"))
    assert body["max_tokens"] == 128


def test_chat_completion_sends_reasoning_effort_when_requested() -> None:
    """Reasoning effort high/medium/low is sent as a top-level field."""

    urlopen = CapturingUrlopen({"choices": [{"message": {"content": "ok"}}]})
    client = ExternalLLMClient(urlopen)

    client.create_chat_completion(
        "https://provider.example/v1",
        "secret",
        ExternalLLMChatRequest(
            model="model-a",
            system_prompt="system",
            user_prompt="user",
            reasoning_effort="low",
        ),
    )

    request, _timeout = urlopen.requests[0]
    assert isinstance(request.data, bytes)
    body = json.loads(request.data.decode("utf-8"))
    assert body["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in body


def test_chat_completion_sends_thinking_disabled_for_off() -> None:
    """The off option maps to chat template thinking disablement."""

    urlopen = CapturingUrlopen({"choices": [{"message": {"content": "ok"}}]})
    client = ExternalLLMClient(urlopen)

    client.create_chat_completion(
        "https://provider.example/v1",
        "secret",
        ExternalLLMChatRequest(
            model="model-a",
            system_prompt="system",
            user_prompt="user",
            reasoning_effort="off",
        ),
    )

    request, _timeout = urlopen.requests[0]
    assert isinstance(request.data, bytes)
    body = json.loads(request.data.decode("utf-8"))
    assert body["chat_template_kwargs"] == {"thinking": False}
    assert "reasoning_effort" not in body


def test_chat_completion_sends_image_content_when_image_is_present() -> None:
    """Vision requests use OpenAI-compatible multimodal user content."""

    urlopen = CapturingUrlopen({"choices": [{"message": {"content": "caption"}}]})
    client = ExternalLLMClient(urlopen)

    client.create_chat_completion(
        "https://provider.example/v1",
        "secret",
        ExternalLLMChatRequest(
            model="model-a",
            system_prompt="system",
            user_prompt="describe this",
            image_data_url="data:image/png;base64,abc",
        ),
    )

    request, _timeout = urlopen.requests[0]
    assert isinstance(request.data, bytes)
    body = json.loads(request.data.decode("utf-8"))
    assert body["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            },
        ],
    }


def test_chat_completion_rejects_malformed_response() -> None:
    """Missing assistant content is reported as a provider response error."""

    client = ExternalLLMClient(CapturingUrlopen({"choices": []}))

    with pytest.raises(ExternalLLMProviderError, match="chat completion"):
        client.create_chat_completion(
            "https://provider.example/v1",
            "secret",
            ExternalLLMChatRequest(
                model="model-a",
                system_prompt="system",
                user_prompt="user",
            ),
        )


def test_transport_failures_become_provider_errors() -> None:
    """URL transport errors are wrapped as provider errors."""

    def failing_urlopen(
        _request: urllib.request.Request,
        _timeout: int,
    ) -> FakeResponse:
        raise urllib.error.URLError("offline")

    client = ExternalLLMClient(failing_urlopen)

    with pytest.raises(ExternalLLMProviderError, match="request failed"):
        client.list_models("https://provider.example/v1", "secret")
