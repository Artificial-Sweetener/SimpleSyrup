# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP route registration for external LLM provider settings."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, cast

from aiohttp import web

from ..domain.external_llm import ExternalLLMConfigError, ExternalLLMProviderError
from ..services.external_llm_prompt_service import ExternalLLMPromptService
from ..shared.logging import get_logger
from .external_llm_keyring import ExternalLLMKeyringError

LOGGER = get_logger(__name__)
EXTERNAL_LLM_SETTINGS_ROUTE = "/simple-syrup/external-llm/settings"
EXTERNAL_LLM_API_KEY_ROUTE = "/simple-syrup/external-llm/api-key"
EXTERNAL_LLM_MODELS_REFRESH_ROUTE = "/simple-syrup/external-llm/models/refresh"

Handler = Callable[[Any], Coroutine[Any, Any, web.Response]]
_REGISTERED_PROMPT_SERVERS: set[int] = set()


class RoutesProtocol(Protocol):
    """Subset of Comfy's route table needed for route registration."""

    def get(self, path: str) -> Callable[[Handler], Handler]:
        """Return a GET route decorator."""

    def post(self, path: str) -> Callable[[Handler], Handler]:
        """Return a POST route decorator."""

    def delete(self, path: str) -> Callable[[Handler], Handler]:
        """Return a DELETE route decorator."""


class PromptServerProtocol(Protocol):
    """Subset of Comfy's PromptServer needed for route registration."""

    routes: RoutesProtocol


class ExternalLLMHandlers:
    """Handle SimpleSyrup external LLM HTTP requests."""

    def __init__(self, service: ExternalLLMPromptService) -> None:
        """Create handlers backed by the external LLM service."""

        self._service = service

    async def get_settings(self, _request: Any) -> web.Response:
        """Return current non-secret external LLM settings."""

        return web.json_response(self._settings_payload())

    async def post_settings(self, request: Any) -> web.Response:
        """Validate and persist non-secret external LLM settings."""

        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ExternalLLMConfigError(
                    "External LLM settings request body must be a JSON object."
                )
            base_url = payload.get("base_url", "")
            default_model = payload.get("default_model", "")
            if not isinstance(base_url, str) or not isinstance(default_model, str):
                raise ExternalLLMConfigError(
                    "External LLM settings require string base_url and "
                    "default_model values."
                )
            self._service.save_config(base_url, default_model)
        except ExternalLLMConfigError as error:
            return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            LOGGER.warning(
                "invalid external llm settings request body",
                extra={"route": EXTERNAL_LLM_SETTINGS_ROUTE, "reason": str(error)},
            )
            return web.json_response(
                {"error": "External LLM settings request body must be valid JSON."},
                status=400,
            )

        return web.json_response(self._settings_payload())

    async def post_api_key(self, request: Any) -> web.Response:
        """Store an external LLM API key without returning it."""

        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ExternalLLMConfigError(
                    "External LLM API key request body must be a JSON object."
                )
            api_key = payload.get("api_key")
            if not isinstance(api_key, str):
                raise ExternalLLMConfigError(
                    "External LLM API key request requires an api_key string."
                )
            await asyncio.to_thread(self._service.save_api_key, api_key)
        except (
            ExternalLLMConfigError,
            ExternalLLMKeyringError,
            ExternalLLMProviderError,
        ) as error:
            return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            LOGGER.warning(
                "invalid external llm api key request body",
                extra={"route": EXTERNAL_LLM_API_KEY_ROUTE, "reason": str(error)},
            )
            return web.json_response(
                {"error": "External LLM API key request body must be valid JSON."},
                status=400,
            )

        return web.json_response(self._settings_payload())

    async def delete_api_key(self, _request: Any) -> web.Response:
        """Delete the configured external LLM API key."""

        try:
            self._service.delete_api_key()
        except ExternalLLMKeyringError as error:
            return web.json_response({"error": str(error)}, status=400)

        return web.json_response(self._settings_payload())

    async def refresh_models(self, _request: Any) -> web.Response:
        """Refresh cached external LLM model ids."""

        try:
            await asyncio.to_thread(self._service.refresh_models)
        except (
            ExternalLLMConfigError,
            ExternalLLMKeyringError,
            ExternalLLMProviderError,
        ) as error:
            return web.json_response({"error": str(error)}, status=400)

        return web.json_response(self._settings_payload())

    def _settings_payload(self) -> dict[str, object]:
        """Return non-secret external LLM settings plus API key presence."""

        return self._service.settings_payload()


def register_external_llm_routes(
    service: ExternalLLMPromptService | None = None,
    prompt_server: PromptServerProtocol | None = None,
) -> bool:
    """Register external LLM routes with Comfy's PromptServer."""

    server_instance = prompt_server or _prompt_server_instance()
    if server_instance is None:
        return False

    server_key = id(server_instance)
    if prompt_server is None and server_key in _REGISTERED_PROMPT_SERVERS:
        return True

    handlers = ExternalLLMHandlers(service or ExternalLLMPromptService())
    server_instance.routes.get(EXTERNAL_LLM_SETTINGS_ROUTE)(handlers.get_settings)
    server_instance.routes.post(EXTERNAL_LLM_SETTINGS_ROUTE)(handlers.post_settings)
    server_instance.routes.post(EXTERNAL_LLM_API_KEY_ROUTE)(handlers.post_api_key)
    server_instance.routes.delete(EXTERNAL_LLM_API_KEY_ROUTE)(handlers.delete_api_key)
    server_instance.routes.post(EXTERNAL_LLM_MODELS_REFRESH_ROUTE)(
        handlers.refresh_models
    )
    if prompt_server is None:
        _REGISTERED_PROMPT_SERVERS.add(server_key)
    return True


def _prompt_server_instance() -> PromptServerProtocol | None:
    """Return Comfy's PromptServer instance when available."""

    try:
        server_module = sys.modules["server"]
        prompt_server = server_module.PromptServer
        instance = prompt_server.instance
    except (KeyError, AttributeError) as error:
        LOGGER.debug(
            "PromptServer unavailable for external LLM routes",
            extra={"reason": str(error)},
        )
        return None
    return cast(PromptServerProtocol, instance)
