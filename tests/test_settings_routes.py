# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SimpleSyrup backend settings routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from aiohttp import web

import simple_syrup.runtime.settings_routes as settings_routes
from simple_syrup.runtime.settings import (
    SimpleSyrupSettings,
    SimpleSyrupSettingsRepository,
)
from simple_syrup.runtime.settings_routes import (
    SETTINGS_ROUTE,
    Handler,
    PromptServerProtocol,
    register_settings_routes,
)


class FakeRoutes:
    """Route table double that records decorated handlers."""

    def __init__(self) -> None:
        """Create an empty fake route table."""

        self.get_handlers: dict[str, Handler] = {}
        self.post_handlers: dict[str, Handler] = {}

    def get(self, path: str) -> Callable[[Handler], Handler]:
        """Record a GET route handler."""

        def decorator(handler: Handler) -> Handler:
            self.get_handlers[path] = handler
            return handler

        return decorator

    def post(self, path: str) -> Callable[[Handler], Handler]:
        """Record a POST route handler."""

        def decorator(handler: Handler) -> Handler:
            self.post_handlers[path] = handler
            return handler

        return decorator


class FakePromptServer:
    """PromptServer double exposing a route table."""

    def __init__(self) -> None:
        """Create a fake PromptServer."""

        self.routes = FakeRoutes()


class FakeRequest:
    """Request double with injectable JSON body behavior."""

    def __init__(self, payload: object | BaseException) -> None:
        """Create a request that returns or raises from `json()`."""

        self._payload = payload

    async def json(self) -> object:
        """Return the configured JSON payload."""

        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


def test_route_registration_records_get_and_post_handlers(tmp_path: Path) -> None:
    """Settings routes register with Comfy's PromptServer routes."""

    prompt_server = FakePromptServer()
    repository = SimpleSyrupSettingsRepository(tmp_path / "settings.json")

    assert register_fake_routes(repository, prompt_server) is True

    assert SETTINGS_ROUTE in prompt_server.routes.get_handlers
    assert SETTINGS_ROUTE in prompt_server.routes.post_handlers


def test_route_registration_is_import_safe_without_prompt_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Comfy PromptServer does not break package import."""

    monkeypatch.setattr(settings_routes, "_prompt_server_instance", lambda: None)

    assert register_settings_routes(prompt_server=None) is False


def test_get_settings_returns_current_settings(tmp_path: Path) -> None:
    """GET returns the current persisted settings payload."""

    prompt_server = FakePromptServer()
    repository = SimpleSyrupSettingsRepository(tmp_path / "settings.json")
    repository.save(SimpleSyrupSettings(show_downloadable_models=False))
    register_fake_routes(repository, prompt_server)

    response = asyncio.run(prompt_server.routes.get_handlers[SETTINGS_ROUTE](object()))

    assert response.status == 200
    assert json.loads(response_text(response)) == {"show_downloadable_models": False}


def test_post_settings_validates_and_persists_payload(tmp_path: Path) -> None:
    """POST validates and saves settings."""

    prompt_server = FakePromptServer()
    repository = SimpleSyrupSettingsRepository(tmp_path / "settings.json")
    register_fake_routes(repository, prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[SETTINGS_ROUTE](
            FakeRequest({"show_downloadable_models": False})
        )
    )

    assert response.status == 200
    assert json.loads(response_text(response)) == {"show_downloadable_models": False}
    assert repository.load().show_downloadable_models is False


def test_post_settings_rejects_non_boolean_payload(tmp_path: Path) -> None:
    """POST rejects malformed setting values."""

    prompt_server = FakePromptServer()
    repository = SimpleSyrupSettingsRepository(tmp_path / "settings.json")
    register_fake_routes(repository, prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[SETTINGS_ROUTE](
            FakeRequest({"show_downloadable_models": "false"})
        )
    )

    assert response.status == 400
    assert "show_downloadable_models" in response_text(response)


def test_post_settings_rejects_invalid_json(tmp_path: Path) -> None:
    """POST rejects bodies that cannot be decoded as JSON."""

    prompt_server = FakePromptServer()
    repository = SimpleSyrupSettingsRepository(tmp_path / "settings.json")
    register_fake_routes(repository, prompt_server)

    response = asyncio.run(
        prompt_server.routes.post_handlers[SETTINGS_ROUTE](
            FakeRequest(ValueError("bad json"))
        )
    )

    assert response.status == 400
    assert "valid JSON" in response_text(response)


def register_fake_routes(
    repository: SimpleSyrupSettingsRepository,
    prompt_server: FakePromptServer,
) -> bool:
    """Register routes against a fake PromptServer with structural typing."""

    return register_settings_routes(
        repository,
        cast(PromptServerProtocol, prompt_server),
    )


def response_text(response: web.Response) -> str:
    """Return response text after asserting aiohttp populated it."""

    assert response.text is not None
    return response.text
