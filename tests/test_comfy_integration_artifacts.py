# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test atomic managed-run evidence persistence and finalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts


def _record(artifacts: IntegrationArtifacts) -> JsonObject:
    """Read and narrow the authoritative run record."""

    decoded: object = json.loads((artifacts.root / "run.json").read_text("utf-8"))
    assert isinstance(decoded, dict)
    return cast(JsonObject, decoded)


def test_success_artifacts_retain_command_history_image_and_hash(
    tmp_path: Path,
) -> None:
    """Persist the complete evidence set and finalize after cleanup."""

    artifacts = IntegrationArtifacts(tmp_path)
    artifacts.record_started(
        command=("python.exe", "main.py"),
        environment={"platform": "Windows"},
        required_node_ids=frozenset({"SaveImage"}),
        port=8299,
        pid=4242,
    )
    history: JsonObject = {"status": {"completed": True}}
    artifacts.record_success(
        system_stats={"device": "gpu"},
        workflow={"1": {"class_type": "SaveImage", "inputs": {}}},
        history=history,
        prompt_id="prompt-id",
        image_reference=ImageReference("image.png", "suite", "output"),
        image_bytes=b"png-evidence",
    )
    artifacts.record_cleanup(process_running=False, port_available=True)

    record = _record(artifacts)
    assert record["status"] == "completed"
    assert record["parent_pid"] == 4242
    assert record["prompt_id"] == "prompt-id"
    assert isinstance(record["image_sha256"], str)
    assert (artifacts.root / "history.json").is_file()
    assert (artifacts.root / "baseline.png").read_bytes() == b"png-evidence"


def test_cleanup_refuses_to_finalize_a_live_parent(tmp_path: Path) -> None:
    """Keep success unfinalized when the exact parent is still alive."""

    artifacts = IntegrationArtifacts(tmp_path)

    with pytest.raises(RuntimeError, match="remains alive"):
        artifacts.record_cleanup(process_running=True, port_available=True)


def test_cleanup_refuses_to_finalize_an_occupied_port(tmp_path: Path) -> None:
    """Keep success unfinalized while a listener remains on the managed port."""

    artifacts = IntegrationArtifacts(tmp_path)

    with pytest.raises(RuntimeError, match="remains occupied"):
        artifacts.record_cleanup(process_running=False, port_available=False)


def test_failure_preserves_exception_type_and_message(tmp_path: Path) -> None:
    """Record an actionable terminal failure state."""

    artifacts = IntegrationArtifacts(tmp_path)
    artifacts.record_failure(ValueError("bad workflow"))

    record = _record(artifacts)
    assert record["status"] == "failed"
    assert record["error"] == "ValueError: bad workflow"
