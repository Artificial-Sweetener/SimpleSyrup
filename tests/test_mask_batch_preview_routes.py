# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for channel-accurate authored-mask preview routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest
from aiohttp import web
from PIL import Image

import simple_syrup.runtime.mask_batch_preview_routes as preview_routes
from simple_syrup.runtime.mask_batch_preview_routes import (
    MASK_BATCH_PREVIEW_ROUTE,
    Handler,
    MaskBatchLoaderProtocol,
    MaskBatchPreviewHandlers,
    MaskBatchPreviewRendererProtocol,
    NativeMaskBatchPreviewRenderer,
    PromptServerProtocol,
    register_mask_batch_preview_routes,
)
from simple_syrup.services.load_mask_batch_service import LoadMaskBatchService


class FakeRoutes:
    """Route table double that records decorated POST handlers."""

    def __init__(self) -> None:
        """Create an empty fake route table."""

        self.post_handlers: dict[str, Handler] = {}

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


class FakeMaskBatchLoader:
    """Record preview loads and return recognizable mask objects."""

    def __init__(self, failure: ValueError | None = None) -> None:
        """Create a loader with an optional validation failure."""

        self.failure = failure
        self.loaded: list[tuple[tuple[str, ...], str]] = []
        self.masks = (object(), object())

    def load_each(self, files: Sequence[str], channel: str) -> Sequence[object]:
        """Record the ordered request and return configured masks."""

        if self.failure is not None:
            raise self.failure
        self.loaded.append((tuple(files), channel))
        return self.masks


class FakeMaskBatchPreviewRenderer:
    """Return a recognizable native-preview response."""

    def __init__(self) -> None:
        """Create a renderer that records the loaded batch."""

        self.rendered: list[tuple[object, ...]] = []
        self.payload: dict[str, object] = {
            "images": [
                {
                    "filename": "ComfyUI_temp_mask.png",
                    "subfolder": "",
                    "type": "temp",
                }
            ],
            "animated": [False],
        }

    def render(self, masks: Sequence[object]) -> dict[str, object]:
        """Record and return the native preview payload."""

        self.rendered.append(tuple(masks))
        return self.payload


def test_preview_route_loads_selected_channel_and_returns_native_output() -> None:
    """The preview uses the same ordered files and channel as node execution."""

    prompt_server = FakePromptServer()
    loader = FakeMaskBatchLoader()
    renderer = FakeMaskBatchPreviewRenderer()
    register_fake_routes(prompt_server, loader, renderer)

    response = asyncio.run(
        prompt_server.routes.post_handlers[MASK_BATCH_PREVIEW_ROUTE](
            FakeRequest(
                {
                    "files": ["characters/left.png", "characters/right.png"],
                    "channel": "green",
                }
            )
        )
    )

    assert response.status == 200
    assert json.loads(response_text(response)) == renderer.payload
    assert loader.loaded == [(("characters/left.png", "characters/right.png"), "green")]
    assert renderer.rendered == [loader.masks]


def test_preview_route_uses_real_native_loader_and_renderer_on_synthetic_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic masks preview through the same native path as node execution."""

    input_directory = tmp_path / "input"
    preview_directory = tmp_path / "preview"
    input_directory.mkdir()
    preview_directory.mkdir()
    paths = {
        "left.png": input_directory / "left.png",
        "right.png": input_directory / "right.png",
    }
    Image.new("RGBA", (8, 6), color=(255, 255, 255, 0)).save(paths["left.png"])
    Image.new("RGBA", (8, 6), color=(0, 0, 0, 255)).save(paths["right.png"])

    folder_paths = import_module("folder_paths")
    monkeypatch.setattr(
        folder_paths,
        "exists_annotated_filepath",
        lambda value: value in paths,
    )
    monkeypatch.setattr(
        folder_paths,
        "get_annotated_filepath",
        lambda value: str(paths[value]),
    )
    monkeypatch.setattr(
        folder_paths,
        "get_temp_directory",
        lambda: str(preview_directory),
    )

    handler = MaskBatchPreviewHandlers(
        LoadMaskBatchService(),
        NativeMaskBatchPreviewRenderer(),
    )
    response = asyncio.run(
        handler.post_preview(
            FakeRequest({"files": ["left.png", "right.png"], "channel": "alpha"})
        )
    )

    payload = json.loads(response_text(response))
    assert response.status == 200
    assert len(payload["images"]) == 2
    assert payload["animated"] == [False]
    for image in payload["images"]:
        assert image["type"] == "temp"
        preview_path = preview_directory / image["subfolder"] / image["filename"]
        with Image.open(preview_path) as preview:
            assert preview.size == (8, 6)


def test_preview_route_preserves_rgb_geometry_for_missing_alpha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An RGB file previews as source-sized zero coverage for the alpha channel."""

    input_directory = tmp_path / "input"
    preview_directory = tmp_path / "preview"
    input_directory.mkdir()
    preview_directory.mkdir()
    mask_path = input_directory / "rgb.png"
    Image.new("RGB", (9, 7), color=(255, 255, 255)).save(mask_path)

    folder_paths = import_module("folder_paths")
    monkeypatch.setattr(
        folder_paths,
        "exists_annotated_filepath",
        lambda value: value == "rgb.png",
    )
    monkeypatch.setattr(
        folder_paths,
        "get_annotated_filepath",
        lambda value: str(mask_path),
    )
    monkeypatch.setattr(
        folder_paths,
        "get_temp_directory",
        lambda: str(preview_directory),
    )

    handler = MaskBatchPreviewHandlers(
        LoadMaskBatchService(),
        NativeMaskBatchPreviewRenderer(),
    )
    response = asyncio.run(
        handler.post_preview(FakeRequest({"files": ["rgb.png"], "channel": "alpha"}))
    )

    payload = json.loads(response_text(response))
    assert response.status == 200
    preview_record = payload["images"][0]
    preview_path = (
        preview_directory / preview_record["subfolder"] / preview_record["filename"]
    )
    with Image.open(preview_path) as preview:
        assert preview.size == (9, 7)
        assert preview.convert("L").getextrema() == (0, 0)


def test_preview_route_renders_mixed_dimensions_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview every selected mask even when execution cannot form one batch."""

    input_directory = tmp_path / "input"
    preview_directory = tmp_path / "preview"
    input_directory.mkdir()
    preview_directory.mkdir()
    paths = {
        "small.png": input_directory / "small.png",
        "large.png": input_directory / "large.png",
    }
    Image.new("RGBA", (8, 6), color=(255, 255, 255, 0)).save(paths["small.png"])
    Image.new("RGBA", (12, 10), color=(0, 0, 0, 255)).save(paths["large.png"])

    folder_paths = import_module("folder_paths")
    monkeypatch.setattr(
        folder_paths,
        "exists_annotated_filepath",
        lambda value: value in paths,
    )
    monkeypatch.setattr(
        folder_paths,
        "get_annotated_filepath",
        lambda value: str(paths[value]),
    )
    monkeypatch.setattr(
        folder_paths,
        "get_temp_directory",
        lambda: str(preview_directory),
    )

    service = LoadMaskBatchService()
    handler = MaskBatchPreviewHandlers(
        service,
        NativeMaskBatchPreviewRenderer(),
    )
    response = asyncio.run(
        handler.post_preview(
            FakeRequest({"files": ["small.png", "large.png"], "channel": "red"})
        )
    )

    payload = json.loads(response_text(response))
    assert response.status == 200
    assert len(payload["images"]) == 2
    preview_sizes = []
    for image in payload["images"]:
        preview_path = preview_directory / image["subfolder"] / image["filename"]
        with Image.open(preview_path) as preview:
            preview_sizes.append(preview.size)
    assert preview_sizes == [(8, 6), (12, 10)]
    with pytest.raises(ValueError, match="identical dimensions"):
        service.load(["small.png", "large.png"], "red")


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"channel": "alpha"}, "files"),
        ({"files": [], "channel": "alpha"}, "at least one"),
        ({"files": ["mask.png"], "channel": 1}, "channel"),
    ],
)
def test_preview_route_rejects_malformed_payloads(
    payload: object,
    message: str,
) -> None:
    """Malformed preview requests fail before filesystem loading."""

    prompt_server = FakePromptServer()
    loader = FakeMaskBatchLoader()
    register_fake_routes(prompt_server, loader, FakeMaskBatchPreviewRenderer())

    response = asyncio.run(
        prompt_server.routes.post_handlers[MASK_BATCH_PREVIEW_ROUTE](
            FakeRequest(payload)
        )
    )

    assert response.status == 400
    assert message in response_text(response)
    assert loader.loaded == []


def test_preview_route_surfaces_loader_validation_errors() -> None:
    """Native path, channel, and shape errors remain actionable to the client."""

    prompt_server = FakePromptServer()
    loader = FakeMaskBatchLoader(ValueError("mask dimensions do not match"))
    register_fake_routes(prompt_server, loader, FakeMaskBatchPreviewRenderer())

    response = asyncio.run(
        prompt_server.routes.post_handlers[MASK_BATCH_PREVIEW_ROUTE](
            FakeRequest({"files": ["mask.png"], "channel": "alpha"})
        )
    )

    assert response.status == 400
    assert json.loads(response_text(response)) == {
        "error": "mask dimensions do not match"
    }


def test_preview_route_rejects_invalid_json() -> None:
    """Bodies that cannot be decoded never reach the mask loader."""

    prompt_server = FakePromptServer()
    loader = FakeMaskBatchLoader()
    register_fake_routes(prompt_server, loader, FakeMaskBatchPreviewRenderer())

    response = asyncio.run(
        prompt_server.routes.post_handlers[MASK_BATCH_PREVIEW_ROUTE](
            FakeRequest(ValueError("bad json"))
        )
    )

    assert response.status == 400
    assert "valid JSON" in response_text(response)
    assert loader.loaded == []


def test_route_registration_is_import_safe_without_prompt_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Comfy PromptServer does not import or initialize mask services."""

    monkeypatch.setattr(preview_routes, "_prompt_server_instance", lambda: None)

    assert register_mask_batch_preview_routes(prompt_server=None) is False


def register_fake_routes(
    prompt_server: FakePromptServer,
    loader: FakeMaskBatchLoader,
    renderer: FakeMaskBatchPreviewRenderer,
) -> bool:
    """Register preview routes against structurally typed test doubles."""

    return register_mask_batch_preview_routes(
        loader=cast(MaskBatchLoaderProtocol, loader),
        renderer=cast(MaskBatchPreviewRendererProtocol, renderer),
        prompt_server=cast(PromptServerProtocol, prompt_server),
    )


def response_text(response: web.Response) -> str:
    """Return response text after asserting aiohttp populated it."""

    assert response.text is not None
    return response.text
