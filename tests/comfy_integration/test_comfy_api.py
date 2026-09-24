# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify bounded loopback-only Comfy API access."""

from __future__ import annotations

import http.client
import json
import urllib.request
from collections import deque
from typing import cast

import pytest

from tools.comfy_api import ImageReference, LoopbackComfyClient


class _Response:
    """Provide the urllib response surface used by the client."""

    def __init__(self, body: bytes) -> None:
        """Store one response body."""

        self._body = body

    def __enter__(self) -> _Response:
        """Enter the fake response context."""

        return self

    def __exit__(self, *args: object) -> None:
        """Exit the fake response context."""

    def read(self) -> bytes:
        """Return the configured response body."""

        return self._body


def test_client_verifies_submits_polls_and_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete HTTP surface with deterministic responses."""

    required = frozenset({"Required.Node"})
    responses = deque(
        [
            json.dumps({"system": {}, "devices": [{}]}).encode(),
            json.dumps({"Required.Node": {}}).encode(),
            json.dumps(
                {
                    "Required.Node": {
                        "name": "Required.Node",
                        "display_name": "Required Node",
                    }
                }
            ).encode(),
            json.dumps({"prompt_id": "prompt-1", "node_errors": {}}).encode(),
            json.dumps(
                {
                    "prompt-1": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {},
                    }
                }
            ).encode(),
            b"image-bytes",
        ]
    )
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        """Record each bounded request and return the next response."""

        assert timeout > 0
        requests.append(request)
        return _Response(responses.popleft())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = LoopbackComfyClient("http://127.0.0.1:8297", poll_interval=0.001)

    assert client.verify_server(required)["system"] == {}
    assert client.node_metadata("Required.Node")["display_name"] == "Required Node"
    prompt_id = client.submit({"1": {"class_type": "Test", "inputs": {}}})
    history = client.wait_for_history(prompt_id, timeout=1.0)
    image = client.download_image(ImageReference("a.png", "bench", "output"))

    assert prompt_id == "prompt-1"
    assert history["outputs"] == {}
    assert image == b"image-bytes"
    assert [request.get_method() for request in requests] == [
        "GET",
        "GET",
        "GET",
        "POST",
        "GET",
        "GET",
    ]


def test_client_rejects_missing_or_malformed_node_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when live node metadata cannot prove one exact contract."""

    responses = deque([b"{}", b'{"Target.Node": []}'])
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response(responses.popleft()),
    )
    client = LoopbackComfyClient("http://127.0.0.1:8297")

    with pytest.raises(ValueError, match="Target.Node"):
        client.node_metadata("Target.Node")
    with pytest.raises(TypeError, match="metadata"):
        client.node_metadata("Target.Node")


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8297",
        "http://0.0.0.0:8297",
        "http://example.com:8297",
        "http://127.0.0.1:8297/path",
    ],
)
def test_client_rejects_non_loopback_or_ambiguous_urls(url: str) -> None:
    """Keep benchmark traffic bound to one explicit loopback HTTP endpoint."""

    with pytest.raises(ValueError, match="loopback|path"):
        LoopbackComfyClient(url)


def test_client_reports_missing_caller_required_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep workflow-specific node policy with the calling application."""

    responses = deque([b"{}", b"{}"])
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response(responses.popleft()),
    )
    client = LoopbackComfyClient("http://127.0.0.1:8297")

    with pytest.raises(ValueError, match="Required.Node"):
        client.verify_server({"Required.Node"})


def test_client_normalizes_transient_malformed_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let readiness retry a startup connection with no valid HTTP response yet."""

    def malformed_status(
        request: urllib.request.Request,
        timeout: float,
    ) -> _Response:
        """Raise the protocol error observed during managed startup."""

        del request, timeout
        raise http.client.BadStatusLine("GET /system_stats HTTP/1.1")

    monkeypatch.setattr(urllib.request, "urlopen", malformed_status)
    client = LoopbackComfyClient("http://127.0.0.1:8297")

    with pytest.raises(ConnectionError, match="system_stats"):
        client.verify_server(set())


def test_client_returns_terminal_error_history_without_waiting_for_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface Comfy errors whose completed flag remains false."""

    body = json.dumps(
        {
            "prompt-1": {
                "status": {
                    "completed": False,
                    "status_str": "error",
                    "messages": [["execution_error", {"node_id": "5"}]],
                },
                "outputs": {},
            }
        }
    ).encode()
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response(body),
    )
    client = LoopbackComfyClient("http://127.0.0.1:8297", poll_interval=0.001)

    history = client.wait_for_history("prompt-1", timeout=1.0)

    assert cast(dict[str, object], history["status"])["status_str"] == "error"
