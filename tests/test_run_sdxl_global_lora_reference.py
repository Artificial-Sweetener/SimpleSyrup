# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify one-case native global-LoRA orchestration and exact cleanup."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pytest import MonkeyPatch
from sdxl_visual_test_inventory import visual_inventory

import tools.run_sdxl_global_lora_reference as runner
from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_global_lora_reference.cases import NativeGlobalLoraCase


class _FakeClient:
    """Return one successful anonymous native image."""

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Retain that one complete prompt reached the managed boundary."""

        assert prompt
        return "prompt"

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Return one successful terminal history."""

        assert prompt_id == "prompt"
        assert timeout == 30.0
        return {"status": {"status_str": "success", "completed": True}}

    def download_image(self, reference: ImageReference) -> bytes:
        """Return one valid 1024 output for the requested image."""

        assert reference.filename == "output.png"
        return _image_bytes()


class _FakeManagedServer:
    """Expose one stopped loopback process through the context contract."""

    def __init__(self, **values: object) -> None:
        """Require an isolated native-Comfy launch."""

        assert values["launch_arguments"] == (
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes",
            "substitute-backend",
        )
        self.running = SimpleNamespace(
            system_stats={"device": "test"},
            client=_FakeClient(),
            port=54321,
            process=SimpleNamespace(is_running=False),
        )

    def __enter__(self) -> object:
        """Return one fake running process."""

        return self.running

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Leave the fake process stopped."""

        del exc_type, exc, traceback


def test_execute_one_case_preserves_result_and_cleans_links(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Finalize only after the server and exact model links are clean."""

    inventory = visual_inventory(tmp_path)
    model_root = tmp_path / "models"
    (model_root / "checkpoints").mkdir(parents=True)
    (model_root / "loras").mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "resolve_active_comfy_model_root",
        lambda _root: model_root,
    )
    monkeypatch.setattr(runner, "ManagedComfyServer", _FakeManagedServer)
    monkeypatch.setattr(runner, "is_loopback_port_available", lambda _port: True)
    monkeypatch.setattr(
        runner,
        "extract_saved_image",
        lambda _history, _node: ImageReference("output.png", "", "output"),
    )
    artifacts = IntegrationArtifacts(tmp_path / "artifacts")

    result = runner.execute_native_global_lora_reference(
        artifacts,
        case=NativeGlobalLoraCase.TRIGGER_CONTROL,
        inventory=inventory,
        comfy_root=tmp_path,
        readiness_timeout=10.0,
        prompt_timeout=30.0,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    run = json.loads((artifacts.root / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert not (model_root / "checkpoints" / "simple_syrup_u11").exists()
    assert not (model_root / "loras" / "simple_syrup_u11").exists()


def _image_bytes() -> bytes:
    """Return one anonymous valid native output."""

    image = Image.new("RGB", (1024, 1024), (10, 20, 30))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
