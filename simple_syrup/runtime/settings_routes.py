# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP route registration for SimpleSyrup backend settings."""

from __future__ import annotations

import sys
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, cast

from aiohttp import web

from ..shared.logging import get_logger
from .settings import (
    SimpleSyrupSettings,
    SimpleSyrupSettingsError,
    SimpleSyrupSettingsRepository,
)

LOGGER = get_logger(__name__)
SETTINGS_ROUTE = "/simple-syrup/settings"

Handler = Callable[[Any], Coroutine[Any, Any, web.Response]]
_REGISTERED_PROMPT_SERVERS: set[int] = set()


class RoutesProtocol(Protocol):
    """Subset of Comfy's route table needed for settings registration."""

    def get(self, path: str) -> Callable[[Handler], Handler]:
        """Return a GET route decorator."""

    def post(self, path: str) -> Callable[[Handler], Handler]:
        """Return a POST route decorator."""


class PromptServerProtocol(Protocol):
    """Subset of Comfy's PromptServer needed for route registration."""

    routes: RoutesProtocol


class SettingsHandlers:
    """Handle SimpleSyrup settings HTTP requests."""

    def __init__(self, repository: SimpleSyrupSettingsRepository) -> None:
        """Create handlers backed by the settings repository."""

        self._repository = repository

    async def get_settings(self, _request: Any) -> web.Response:
        """Return current SimpleSyrup settings."""

        return web.json_response(self._repository.load().to_payload())

    async def post_settings(self, request: Any) -> web.Response:
        """Validate and persist SimpleSyrup settings."""

        try:
            payload = await request.json()
            settings = SimpleSyrupSettings.from_payload(payload)
        except SimpleSyrupSettingsError as error:
            return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            LOGGER.warning(
                "invalid settings request body",
                extra={"route": SETTINGS_ROUTE, "reason": str(error)},
            )
            return web.json_response(
                {"error": "SimpleSyrup settings request body must be valid JSON."},
                status=400,
            )

        saved = self._repository.save(settings)
        return web.json_response(saved.to_payload())


def register_settings_routes(
    repository: SimpleSyrupSettingsRepository | None = None,
    prompt_server: PromptServerProtocol | None = None,
) -> bool:
    """Register SimpleSyrup settings routes with Comfy's PromptServer."""

    server_instance = prompt_server or _prompt_server_instance()
    if server_instance is None:
        return False

    server_key = id(server_instance)
    if prompt_server is None and server_key in _REGISTERED_PROMPT_SERVERS:
        return True

    handlers = SettingsHandlers(repository or SimpleSyrupSettingsRepository())
    server_instance.routes.get(SETTINGS_ROUTE)(handlers.get_settings)
    server_instance.routes.post(SETTINGS_ROUTE)(handlers.post_settings)
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
            "PromptServer unavailable for settings routes",
            extra={"reason": str(error)},
        )
        return None
    return cast(PromptServerProtocol, instance)
