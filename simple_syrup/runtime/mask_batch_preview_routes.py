# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP adapter for channel-accurate authored-mask previews."""

from __future__ import annotations

import sys
from collections.abc import Callable, Coroutine, Sequence
from importlib import import_module
from typing import Any, Protocol, cast

from aiohttp import web

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
MASK_BATCH_PREVIEW_ROUTE = "/simple-syrup/mask-batch/preview"

Handler = Callable[[Any], Coroutine[Any, Any, web.Response]]
_REGISTERED_PROMPT_SERVERS: set[int] = set()


class RoutesProtocol(Protocol):
    """Subset of Comfy's route table needed for preview registration."""

    def post(self, path: str) -> Callable[[Handler], Handler]:
        """Return a POST route decorator."""


class PromptServerProtocol(Protocol):
    """Subset of Comfy's PromptServer needed for route registration."""

    routes: RoutesProtocol


class MaskBatchLoaderProtocol(Protocol):
    """Load ordered masks with the same channel semantics as node execution."""

    def load_each(self, files: Sequence[str], channel: str) -> Sequence[object]:
        """Load ordered files independently through the selected mask channel."""


class MaskBatchPreviewRendererProtocol(Protocol):
    """Render loaded masks into Comfy's native execution-output shape."""

    def render(self, masks: Sequence[object]) -> dict[str, object]:
        """Return a JSON-compatible native preview payload."""


class NativeMaskBatchPreviewRenderer:
    """Render mask tensors through Comfy's authoritative PreviewMask helper."""

    def render(self, masks: Sequence[object]) -> dict[str, object]:
        """Save one native temporary preview per independently sized mask."""

        comfy_api: Any = import_module("comfy_api.latest")
        images: list[object] = []
        for mask in masks:
            preview: Any = comfy_api.UI.PreviewMask(mask)
            payload: object = preview.as_dict()
            if not isinstance(payload, dict):
                raise TypeError(
                    "Comfy PreviewMask returned an invalid preview payload."
                )
            preview_images = payload.get("images")
            if not isinstance(preview_images, (list, tuple)):
                raise TypeError("Comfy PreviewMask returned invalid preview images.")
            images.extend(preview_images)
        return {"images": images, "animated": (False,)}


class MaskBatchPreviewHandlers:
    """Handle previews using native mask semantics without batching constraints."""

    def __init__(
        self,
        loader: MaskBatchLoaderProtocol,
        renderer: MaskBatchPreviewRendererProtocol,
    ) -> None:
        """Create handlers with explicit loading and rendering collaborators."""

        self._loader = loader
        self._renderer = renderer

    async def post_preview(self, request: Any) -> web.Response:
        """Return native previews for ordered files and one selected channel."""

        try:
            payload: object = await request.json()
        except Exception as error:
            LOGGER.warning(
                "invalid mask batch preview request body",
                extra={"route": MASK_BATCH_PREVIEW_ROUTE, "reason": str(error)},
            )
            return web.json_response(
                {"error": ("Load Mask Batch preview request body must be valid JSON.")},
                status=400,
            )

        try:
            files, channel = self._request_values(payload)
            masks = self._loader.load_each(files, channel)
        except (TypeError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=400)

        try:
            preview = self._renderer.render(masks)
        except Exception as error:
            LOGGER.exception(
                "mask batch preview rendering failed",
                extra={"route": MASK_BATCH_PREVIEW_ROUTE, "reason": str(error)},
            )
            return web.json_response(
                {"error": "ComfyUI could not render the mask batch preview."},
                status=500,
            )
        return web.json_response(preview)

    def _request_values(self, payload: object) -> tuple[tuple[str, ...], str]:
        """Validate and narrow a JSON preview request."""

        if not isinstance(payload, dict):
            raise TypeError("Load Mask Batch preview payload must be an object.")

        files = payload.get("files")
        if not isinstance(files, list):
            raise TypeError("Load Mask Batch preview files must be a list.")
        if not files:
            raise ValueError("Load Mask Batch preview requires at least one mask file.")
        if any(not isinstance(path, str) or not path for path in files):
            raise TypeError("Load Mask Batch preview files must be non-empty strings.")

        channel = payload.get("channel")
        if not isinstance(channel, str) or not channel:
            raise TypeError(
                "Load Mask Batch preview channel must be a non-empty string."
            )
        return tuple(files), channel


def register_mask_batch_preview_routes(
    loader: MaskBatchLoaderProtocol | None = None,
    renderer: MaskBatchPreviewRendererProtocol | None = None,
    prompt_server: PromptServerProtocol | None = None,
) -> bool:
    """Register mask preview routes with Comfy's PromptServer when available."""

    server_instance = prompt_server or _prompt_server_instance()
    if server_instance is None:
        return False

    server_key = id(server_instance)
    if prompt_server is None and server_key in _REGISTERED_PROMPT_SERVERS:
        return True

    if loader is None:
        from ..services.load_mask_batch_service import LoadMaskBatchService

        loader = LoadMaskBatchService()
    handlers = MaskBatchPreviewHandlers(
        loader,
        renderer or NativeMaskBatchPreviewRenderer(),
    )
    server_instance.routes.post(MASK_BATCH_PREVIEW_ROUTE)(handlers.post_preview)
    if prompt_server is None:
        _REGISTERED_PROMPT_SERVERS.add(server_key)
    return True


def _prompt_server_instance() -> PromptServerProtocol | None:
    """Return Comfy's PromptServer instance without importing it eagerly."""

    try:
        server_module = sys.modules["server"]
        prompt_server = server_module.PromptServer
        instance = prompt_server.instance
    except (KeyError, AttributeError) as error:
        LOGGER.debug(
            "PromptServer unavailable for mask batch preview routes",
            extra={"reason": str(error)},
        )
        return None
    return cast(PromptServerProtocol, instance)
