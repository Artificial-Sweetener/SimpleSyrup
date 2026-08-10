# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose global quant cache status and safe inactive-artifact clearing."""

from __future__ import annotations

import sys
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, cast

from aiohttp import web

from ..services.quant_cache_service import (
    QuantCacheEvictionResult,
    QuantCacheService,
    QuantCacheStatus,
)
from ..services.quantized_model_boundaries import QuantCacheLimitProvider
from .quant_cache_settings import SettingsQuantCacheLimitProvider

QUANT_CACHE_ROUTE = "/simple-syrup/quant-cache"
Handler = Callable[[Any], Coroutine[Any, Any, web.Response]]
_REGISTERED_PROMPT_SERVERS: set[int] = set()


class QuantCacheRoutesProtocol(Protocol):
    """Describe the route decorators used by quant cache endpoints."""

    def get(self, path: str) -> Callable[[Handler], Handler]:
        """Return a GET route decorator."""

    def delete(self, path: str) -> Callable[[Handler], Handler]:
        """Return a DELETE route decorator."""

    def post(self, path: str) -> Callable[[Handler], Handler]:
        """Return a POST route decorator."""


class QuantCachePromptServerProtocol(Protocol):
    """Describe the PromptServer state required by cache routes."""

    routes: QuantCacheRoutesProtocol


class QuantCacheServiceBoundary(Protocol):
    """Describe global cache operations exposed through HTTP."""

    def status(self) -> QuantCacheStatus:
        """Return current cache state."""

    def clear_inactive(self) -> QuantCacheEvictionResult:
        """Remove every inactive managed artifact."""

    def enforce_limit(self, limit_bytes: int) -> QuantCacheEvictionResult:
        """Apply the current global LRU budget."""


class QuantCacheHandlers:
    """Serve global quant cache state and explicit clear requests."""

    def __init__(
        self,
        cache_service: QuantCacheServiceBoundary,
        limit_provider: QuantCacheLimitProvider,
    ) -> None:
        """Create handlers with explicit authoritative collaborators."""

        self._cache_service = cache_service
        self._limit_provider = limit_provider

    async def get_status(self, _request: Any) -> web.Response:
        """Return current global cache usage and configured limit."""

        status = self._cache_service.status()
        return web.json_response(status.to_payload(self._limit_provider.limit_bytes()))

    async def clear_inactive(self, _request: Any) -> web.Response:
        """Clear inactive artifacts and return updated global cache status."""

        eviction = self._cache_service.clear_inactive()
        status = self._cache_service.status()
        payload = status.to_payload(self._limit_provider.limit_bytes())
        payload.update(
            {
                "removed_artifacts": eviction.removed_artifacts,
                "removed_bytes": eviction.removed_bytes,
            }
        )
        return web.json_response(payload)

    async def enforce_limit(self, _request: Any) -> web.Response:
        """Apply the persisted limit and return updated global cache status."""

        limit_bytes = self._limit_provider.limit_bytes()
        eviction = self._cache_service.enforce_limit(limit_bytes)
        status = self._cache_service.status()
        payload = status.to_payload(limit_bytes)
        payload.update(
            {
                "removed_artifacts": eviction.removed_artifacts,
                "removed_bytes": eviction.removed_bytes,
            }
        )
        return web.json_response(payload)


def register_quant_cache_routes(
    cache_service: QuantCacheServiceBoundary | None = None,
    limit_provider: QuantCacheLimitProvider | None = None,
    prompt_server: QuantCachePromptServerProtocol | None = None,
) -> bool:
    """Register global cache routes with ComfyUI when PromptServer is available."""

    server_instance = prompt_server or _prompt_server_instance()
    if server_instance is None:
        return False
    server_key = id(server_instance)
    if prompt_server is None and server_key in _REGISTERED_PROMPT_SERVERS:
        return True
    handlers = QuantCacheHandlers(
        cache_service or QuantCacheService(),
        limit_provider or SettingsQuantCacheLimitProvider(),
    )
    server_instance.routes.get(QUANT_CACHE_ROUTE)(handlers.get_status)
    server_instance.routes.post(QUANT_CACHE_ROUTE)(handlers.enforce_limit)
    server_instance.routes.delete(QUANT_CACHE_ROUTE)(handlers.clear_inactive)
    if prompt_server is None:
        _REGISTERED_PROMPT_SERVERS.add(server_key)
    return True


def _prompt_server_instance() -> QuantCachePromptServerProtocol | None:
    """Return ComfyUI's PromptServer instance when available."""

    try:
        server_module = sys.modules["server"]
        prompt_server = server_module.PromptServer
        instance = prompt_server.instance
    except (KeyError, AttributeError):
        return None
    return cast(QuantCachePromptServerProtocol, instance)
