# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for external LLM settings HTTP routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import cast

from aiohttp import web

from simple_syrup.domain.external_llm import (
    ExternalLLMChatRequest,
    ExternalLLMChatResponse,
    ExternalLLMProviderError,
)
from simple_syrup.runtime.external_llm_routes import (
    EXTERNAL_LLM_API_KEY_ROUTE,
    EXTERNAL_LLM_MODELS_REFRESH_ROUTE,
    EXTERNAL_LLM_SETTINGS_ROUTE,
    Handler,
    PromptServerProtocol,
    register_external_llm_routes,
)
from simple_syrup.runtime.settings import ExternalLLMSettings, SimpleSyrupSettings
from simple_syrup.services.external_llm_prompt_service import ExternalLLMPromptService


class FakeRoutes:
    """Route table double that records handlers."""

    def __init__(self) -> None:
        """Create an empty route table."""

        self.get_handlers: dict[str, Handler] = {}
        self.post_handlers: dict[str, Handler] = {}
        self.delete_handlers: dict[str, Handler] = {}

    def get(self, path: str) -> Callable[[Handler], Handler]:
        """Record GET handlers."""

        def decorator(handler: Handler) -> Handler:
            self.get_handlers[path] = handler
            return handler

        return decorator

    def post(self, path: str) -> Callable[[Handler], Handler]:
        """Record POST handlers."""

        def decorator(handler: Handler) -> Handler:
            self.post_handlers[path] = handler
            return handler

        return decorator

    def delete(self, path: str) -> Callable[[Handler], Handler]:
        """Record DELETE handlers."""

        def decorator(handler: Handler) -> Handler:
            self.delete_handlers[path] = handler
            return handler

        return decorator


class FakePromptServer:
    """PromptServer double exposing route table."""

    def __init__(self) -> None:
        """Create fake prompt server."""

        self.routes = FakeRoutes()


class FakeRequest:
    """Request double with injectable JSON payload."""

    def __init__(self, payload: object) -> None:
        """Store payload."""

        self._payload = payload

    async def json(self) -> object:
        """Return configured payload."""

        return self._payload


class FakeSettingsRepository:
    """In-memory settings repository."""

    def __init__(self, settings: SimpleSyrupSettings) -> None:
        """Store initial settings."""

        self.settings = settings

    def load(self) -> SimpleSyrupSettings:
        """Return current settings."""

        return self.settings

    def save(self, settings: SimpleSyrupSettings) -> SimpleSyrupSettings:
        """Persist settings."""

        self.settings = settings
        return settings


class FakeKeyStore:
    """Credential store double."""

    def __init__(self, api_key: str = "") -> None:
        """Store key."""

        self.api_key = api_key

    def has_api_key(self, _base_url: str) -> bool:
        """Return whether a key exists."""

        return bool(self.api_key)

    def get_api_key(self, _base_url: str) -> str:
        """Return key."""

        return self.api_key

    def save_api_key(self, _base_url: str, api_key: str) -> None:
        """Save key."""

        self.api_key = api_key

    def delete_api_key(self, _base_url: str) -> None:
        """Delete key."""

        self.api_key = ""


class FakeClient:
    """Provider client double."""

    def list_models(self, _base_url: str, _api_key: str) -> tuple[str, ...]:
        """Return fake models."""

        return ("model-a", "model-b")

    def create_chat_completion(
        self,
        _base_url: str,
        _api_key: str,
        _request: ExternalLLMChatRequest,
    ) -> ExternalLLMChatResponse:
        """Return fake content."""

        return ExternalLLMChatResponse("response")


class FailingModelClient(FakeClient):
    """Provider client double that fails model discovery."""

    def list_models(self, _base_url: str, _api_key: str) -> tuple[str, ...]:
        """Raise a provider error during model refresh."""

        raise ExternalLLMProviderError("The external LLM provider rejected /models.")


def test_external_llm_route_registration_records_handlers() -> None:
    """External LLM routes register with Comfy's route table."""

    prompt_server = FakePromptServer()

    assert register_fake_routes(configured_service(), prompt_server) is True

    assert EXTERNAL_LLM_SETTINGS_ROUTE in prompt_server.routes.get_handlers
    assert EXTERNAL_LLM_SETTINGS_ROUTE in prompt_server.routes.post_handlers
    assert EXTERNAL_LLM_API_KEY_ROUTE in prompt_server.routes.post_handlers
    assert EXTERNAL_LLM_API_KEY_ROUTE in prompt_server.routes.delete_handlers
    assert EXTERNAL_LLM_MODELS_REFRESH_ROUTE in prompt_server.routes.post_handlers


def test_get_settings_returns_non_secret_payload() -> None:
    """GET returns endpoint settings and key presence without the key."""

    prompt_server = FakePromptServer()
    register_fake_routes(configured_service(api_key="secret"), prompt_server)

    response = asyncio.run(
        prompt_server.routes.get_handlers[EXTERNAL_LLM_SETTINGS_ROUTE](object())
    )

    payload = json.loads(response_text(response))
    assert response.status == 200
    assert payload == {
        "base_url": "https://provider.example/v1",
        "cached_models": ["model-a"],
        "default_model": "model-a",
        "has_api_key": True,
    }
    assert "secret" not in response_text(response)


def test_post_settings_validates_and_saves_payload() -> None:
    """POST settings persists non-secret endpoint config."""

    prompt_server = FakePromptServer()
    service = ExternalLLMPromptService(
        FakeSettingsRepository(SimpleSyrupSettings()),
        FakeKeyStore(),
        FakeClient(),
    )
    register_fake_routes(service, prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[EXTERNAL_LLM_SETTINGS_ROUTE](
            FakeRequest(
                {
                    "base_url": "https://provider.example/v1/",
                    "default_model": "",
                }
            )
        )
    )

    payload = json.loads(response_text(response))
    assert response.status == 200
    assert payload["base_url"] == "https://provider.example/v1"


def test_post_settings_refreshes_models_when_key_exists() -> None:
    """Saving endpoint settings updates model choices when credentials exist."""

    prompt_server = FakePromptServer()
    service = ExternalLLMPromptService(
        FakeSettingsRepository(
            SimpleSyrupSettings(
                external_llm=ExternalLLMSettings(
                    base_url="https://provider.example/v1",
                    cached_models=("old-model",),
                    default_model="old-model",
                )
            )
        ),
        FakeKeyStore("secret"),
        FakeClient(),
    )
    register_fake_routes(service, prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[EXTERNAL_LLM_SETTINGS_ROUTE](
            FakeRequest(
                {
                    "base_url": "https://provider.example/v1",
                    "default_model": "",
                }
            )
        )
    )

    payload = json.loads(response_text(response))
    assert response.status == 200
    assert payload["cached_models"] == ["model-a", "model-b"]


def test_post_api_key_stores_key_and_refreshes_models() -> None:
    """API key route stores the key without returning it."""

    prompt_server = FakePromptServer()
    register_fake_routes(configured_service(), prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[EXTERNAL_LLM_API_KEY_ROUTE](
            FakeRequest({"api_key": "secret"})
        )
    )

    payload = json.loads(response_text(response))
    assert response.status == 200
    assert payload["has_api_key"] is True
    assert payload["cached_models"] == ["model-a", "model-b"]
    assert "secret" not in response_text(response)


def test_post_api_key_succeeds_when_model_refresh_fails() -> None:
    """API key storage still succeeds if immediate model discovery fails."""

    prompt_server = FakePromptServer()
    service = ExternalLLMPromptService(
        FakeSettingsRepository(
            SimpleSyrupSettings(
                external_llm=ExternalLLMSettings(
                    base_url="https://provider.example/v1",
                )
            )
        ),
        FakeKeyStore(),
        FailingModelClient(),
    )
    register_fake_routes(service, prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[EXTERNAL_LLM_API_KEY_ROUTE](
            FakeRequest({"api_key": "secret"})
        )
    )

    payload = json.loads(response_text(response))
    assert response.status == 200
    assert payload["has_api_key"] is True
    assert payload["cached_models"] == []
    assert "secret" not in response_text(response)


def test_delete_api_key_removes_key_presence() -> None:
    """DELETE removes the stored API key."""

    prompt_server = FakePromptServer()
    register_fake_routes(configured_service(api_key="secret"), prompt_server)

    response = asyncio.run(
        prompt_server.routes.delete_handlers[EXTERNAL_LLM_API_KEY_ROUTE](object())
    )

    assert response.status == 200
    assert json.loads(response_text(response))["has_api_key"] is False


def test_refresh_models_updates_cache_when_configured() -> None:
    """Refresh route updates cached model ids."""

    prompt_server = FakePromptServer()
    register_fake_routes(configured_service(api_key="secret"), prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[EXTERNAL_LLM_MODELS_REFRESH_ROUTE](object())
    )

    assert response.status == 200
    assert json.loads(response_text(response))["cached_models"] == [
        "model-a",
        "model-b",
    ]


def test_refresh_models_skips_when_unconfigured() -> None:
    """Refresh route succeeds without endpoint credentials."""

    prompt_server = FakePromptServer()
    service = ExternalLLMPromptService(
        FakeSettingsRepository(SimpleSyrupSettings()),
        FakeKeyStore(),
        FakeClient(),
    )
    register_fake_routes(service, prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[EXTERNAL_LLM_MODELS_REFRESH_ROUTE](object())
    )

    assert response.status == 200
    assert json.loads(response_text(response))["cached_models"] == []


def configured_service(api_key: str = "") -> ExternalLLMPromptService:
    """Create a configured route service."""

    return ExternalLLMPromptService(
        FakeSettingsRepository(
            SimpleSyrupSettings(
                external_llm=ExternalLLMSettings(
                    base_url="https://provider.example/v1",
                    cached_models=("model-a",),
                    default_model="model-a",
                )
            )
        ),
        FakeKeyStore(api_key),
        FakeClient(),
    )


def register_fake_routes(
    service: ExternalLLMPromptService,
    prompt_server: FakePromptServer,
) -> bool:
    """Register routes against a fake server."""

    return register_external_llm_routes(
        service,
        cast(PromptServerProtocol, prompt_server),
    )


def response_text(response: web.Response) -> str:
    """Return response text after asserting it exists."""

    assert response.text is not None
    return response.text
