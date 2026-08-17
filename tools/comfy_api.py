# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provide bounded loopback-only access to an existing ComfyUI process."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from typing import TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class ImageReference:
    """Identify one image exposed by Comfy's view endpoint."""

    filename: str
    subfolder: str
    output_type: str


class LoopbackComfyClient:
    """Submit prompts to one pre-existing loopback Comfy server."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> None:
        """Validate the endpoint and initialize bounded request controls."""

        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Comfy URL must use HTTP on loopback.")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Comfy URL must not include a path or query.")
        if request_timeout <= 0 or poll_interval <= 0:
            raise ValueError("Comfy client timeouts must be positive.")
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._poll_interval = poll_interval
        self._client_id = str(uuid.uuid4())

    @property
    def websocket_url(self) -> str:
        """Return the matching loopback WebSocket endpoint for this session."""

        parsed = urllib.parse.urlparse(self._base_url)
        host = parsed.netloc
        query = urllib.parse.urlencode({"clientId": self._client_id})
        return f"ws://{host}/ws?{query}"

    def verify_server(self, required_node_ids: Collection[str]) -> JsonObject:
        """Verify readiness and the caller's required node contracts."""

        stats = self._get_json("/system_stats")
        object_info = self._get_json("/object_info")
        missing = sorted(set(required_node_ids) - set(object_info))
        if missing:
            raise ValueError(f"ComfyUI is missing required nodes: {missing}.")
        return stats

    def node_metadata(self, node_id: str) -> JsonObject:
        """Return live metadata for one exact node or fail closed."""

        if not node_id:
            raise ValueError("Comfy node ID must not be empty.")
        encoded = urllib.parse.quote(node_id, safe="")
        object_info = self._get_json(f"/object_info/{encoded}")
        metadata = object_info.get(node_id)
        if metadata is None:
            raise ValueError(f"ComfyUI is missing required node metadata: {node_id}.")
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) for key in metadata
        ):
            raise TypeError(f"ComfyUI metadata for {node_id!r} must be a JSON object.")
        return cast(JsonObject, metadata)

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Submit one API-format prompt and return its server prompt ID."""

        response = self._request_json(
            "/prompt",
            method="POST",
            payload={"prompt": prompt, "client_id": self._client_id},
        )
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            node_errors = response.get("node_errors")
            raise ValueError(f"ComfyUI rejected prompt: {node_errors!r}.")
        return prompt_id

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Poll bounded history until the submitted prompt completes."""

        if timeout <= 0:
            raise ValueError("Comfy history timeout must be positive.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = self._get_json(f"/history/{urllib.parse.quote(prompt_id)}")
            record = history.get(prompt_id)
            if isinstance(record, dict):
                parsed = cast(JsonObject, record)
                status = parsed.get("status")
                if isinstance(status, dict):
                    status_text = status.get("status_str")
                    if status.get("completed") is True or status_text == "error":
                        return parsed
            time.sleep(self._poll_interval)
        raise TimeoutError(
            f"ComfyUI prompt did not complete within {timeout}s: {prompt_id}."
        )

    def download_image(self, reference: ImageReference) -> bytes:
        """Download one exact image reference returned by prompt history."""

        query = urllib.parse.urlencode(
            {
                "filename": reference.filename,
                "subfolder": reference.subfolder,
                "type": reference.output_type,
            }
        )
        request = urllib.request.Request(f"{self._base_url}/view?{query}")
        try:
            with urllib.request.urlopen(
                request, timeout=self._request_timeout
            ) as response:
                result = response.read()
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            raise ConnectionError(
                f"Unable to download Comfy image {reference.filename!r}: {error}"
            ) from error
        if not isinstance(result, bytes):
            raise TypeError("ComfyUI image response must contain bytes.")
        return result

    def _get_json(self, route: str) -> JsonObject:
        """Read one JSON object from a Comfy endpoint."""

        return self._request_json(route, method="GET", payload=None)

    def _request_json(
        self,
        route: str,
        *,
        method: str,
        payload: object | None,
    ) -> JsonObject:
        """Execute one bounded request and narrow its JSON response."""

        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self._base_url}{route}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._request_timeout
            ) as response:
                decoded: object = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise ValueError(
                f"ComfyUI rejected request {route!r} with HTTP {error.code}: {body}"
            ) from error
        except (
            OSError,
            urllib.error.URLError,
            http.client.HTTPException,
            json.JSONDecodeError,
        ) as error:
            raise ConnectionError(
                f"ComfyUI request failed for {route!r}: {error}"
            ) from error
        if not isinstance(decoded, dict) or not all(
            isinstance(key, str) for key in decoded
        ):
            raise TypeError(f"ComfyUI response from {route!r} must be a JSON object.")
        return cast(JsonObject, decoded)
