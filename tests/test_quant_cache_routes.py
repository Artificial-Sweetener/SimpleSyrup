# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for global quant cache settings-menu backend routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import cast

from aiohttp import web

from simple_syrup.runtime.quant_cache_routes import (
    QUANT_CACHE_ROUTE,
    Handler,
    QuantCachePromptServerProtocol,
    register_quant_cache_routes,
)
from simple_syrup.services.quant_cache_service import (
    QuantCacheEvictionResult,
    QuantCacheStatus,
)


class FakeRoutes:
    """Record GET and DELETE cache handlers."""

    def __init__(self) -> None:
        """Create empty route maps."""

        self.get_handlers: dict[str, Handler] = {}
        self.post_handlers: dict[str, Handler] = {}
        self.delete_handlers: dict[str, Handler] = {}

    def get(self, path: str) -> Callable[[Handler], Handler]:
        """Record a GET handler decorator."""

        def decorator(handler: Handler) -> Handler:
            self.get_handlers[path] = handler
            return handler

        return decorator

    def delete(self, path: str) -> Callable[[Handler], Handler]:
        """Record a DELETE handler decorator."""

        def decorator(handler: Handler) -> Handler:
            self.delete_handlers[path] = handler
            return handler

        return decorator

    def post(self, path: str) -> Callable[[Handler], Handler]:
        """Record a POST handler decorator."""

        def decorator(handler: Handler) -> Handler:
            self.post_handlers[path] = handler
            return handler

        return decorator


class FakePromptServer:
    """Expose the fake cache route table."""

    def __init__(self) -> None:
        """Create one route table."""

        self.routes = FakeRoutes()


class FakeCacheService:
    """Return deterministic status and clear results."""

    def __init__(self) -> None:
        """Create a service with one visible artifact."""

        self.cleared = False

    def status(self) -> QuantCacheStatus:
        """Return current fake cache state."""

        return QuantCacheStatus(
            path="models/SyrupQuants",
            usage_bytes=0 if self.cleared else 1024,
            artifact_count=0 if self.cleared else 1,
            active_artifact_count=0,
        )

    def clear_inactive(self) -> QuantCacheEvictionResult:
        """Record clearing and return its result."""

        self.cleared = True
        return QuantCacheEvictionResult(1, 1024, 0)

    def enforce_limit(self, limit_bytes: int) -> QuantCacheEvictionResult:
        """Record enforcement while leaving the tiny artifact within budget."""

        assert limit_bytes == 20 * 1024**3
        return QuantCacheEvictionResult(0, 0, 1024)


class FakeLimitProvider:
    """Return the configured 20 GiB byte limit."""

    def limit_bytes(self) -> int:
        """Return the deterministic limit."""

        return 20 * 1024**3


def test_routes_report_and_clear_the_global_cache() -> None:
    """Settings UI endpoints expose location, limit, usage, and clear results."""

    server = FakePromptServer()
    cache_service = FakeCacheService()
    registered = register_quant_cache_routes(
        cache_service=cache_service,
        limit_provider=FakeLimitProvider(),
        prompt_server=cast(QuantCachePromptServerProtocol, server),
    )

    assert registered is True
    get_response = asyncio.run(server.routes.get_handlers[QUANT_CACHE_ROUTE](object()))
    post_response = asyncio.run(
        server.routes.post_handlers[QUANT_CACHE_ROUTE](object())
    )
    delete_response = asyncio.run(
        server.routes.delete_handlers[QUANT_CACHE_ROUTE](object())
    )

    assert _payload(get_response) == {
        "path": "models/SyrupQuants",
        "usage_bytes": 1024,
        "limit_bytes": 20 * 1024**3,
        "artifact_count": 1,
        "active_artifact_count": 0,
    }
    assert _payload(delete_response) == {
        "path": "models/SyrupQuants",
        "usage_bytes": 0,
        "limit_bytes": 20 * 1024**3,
        "artifact_count": 0,
        "active_artifact_count": 0,
        "removed_artifacts": 1,
        "removed_bytes": 1024,
    }
    assert _payload(post_response)["removed_artifacts"] == 0


def _payload(response: web.Response) -> dict[str, object]:
    """Decode a JSON response created by aiohttp helpers."""

    assert response.text is not None
    payload = json.loads(response.text)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)
