# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test the thin managed-baseline execution coordinator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import tools.run_managed_comfy_baseline as runner
from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.run_managed_comfy_baseline import execute_baseline


class _FakeProcess:
    """Expose post-context liveness to cleanup verification."""

    is_running = True


class _FakeClient:
    """Implement the baseline coordinator's existing HTTP boundary."""

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Accept the production graph and return a prompt identity."""

        assert prompt["7"]["class_type"] == "SaveImage"
        return "prompt-id"

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Return one completed single-image history record."""

        assert prompt_id == "prompt-id"
        assert timeout == 12.0
        return {
            "status": {"completed": True, "status_str": "success", "messages": []},
            "outputs": {
                "7": {
                    "images": [
                        {
                            "filename": "image.png",
                            "subfolder": "suite",
                            "type": "output",
                        }
                    ]
                }
            },
        }

    def download_image(self, reference: ImageReference) -> bytes:
        """Return deterministic image evidence for the extracted reference."""

        assert reference == ImageReference("image.png", "suite", "output")
        return b"png-evidence"


class _FakeRunning:
    """Provide the ready server state used by the coordinator."""

    client = _FakeClient()
    process = _FakeProcess()
    system_stats: JsonObject = {"system": "ready"}
    port = 8299


class _FakeManagedServer:
    """Model guaranteed cleanup at the context boundary."""

    def __init__(self, **kwargs: object) -> None:
        """Accept and retain no lower-level lifecycle policy."""

        del kwargs

    def __enter__(self) -> _FakeRunning:
        """Return one ready fake server."""

        _FakeRunning.process.is_running = True
        return _FakeRunning()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Mark the exact fake process stopped."""

        del exc_type, exc, traceback
        _FakeRunning.process.is_running = False


def test_execute_baseline_records_success_only_after_context_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coordinate graph execution, evidence capture, and final cleanup status."""

    monkeypatch.setattr(runner, "ManagedComfyServer", _FakeManagedServer)
    monkeypatch.setattr(runner, "is_loopback_port_available", lambda port: port == 8299)
    artifacts = IntegrationArtifacts(tmp_path)

    execute_baseline(
        artifacts,
        comfy_root=Path("<COMFY_ROOT>"),
        readiness_timeout=10.0,
        prompt_timeout=12.0,
    )

    decoded: object = json.loads((artifacts.root / "run.json").read_text("utf-8"))
    assert isinstance(decoded, dict)
    record = cast(dict[str, object], decoded)
    assert record["status"] == "completed"
    assert record["prompt_id"] == "prompt-id"
    assert (artifacts.root / "baseline.png").read_bytes() == b"png-evidence"
