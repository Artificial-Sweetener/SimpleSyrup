# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify global-style backend parity orchestration and cleanup."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pytest import MonkeyPatch
from sdxl_visual_test_inventory import visual_inventory

import tools.run_sdxl_attention_couple_parity as runner
from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts


class _FakeClient:
    """Capture ordered parity graphs and return valid anonymous images."""

    class_sets: list[set[object]] = []

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Require the shared global adapter on each backend graph."""

        classes = {node["class_type"] for node in prompt.values()}
        assert "LoraLoader" in classes
        type(self).class_sets.append(classes)
        return f"prompt-{len(type(self).class_sets)}"

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Return one successful history per ordered prompt."""

        assert prompt_id in {"prompt-1", "prompt-2"}
        assert timeout == 30.0
        return {"status": {"status_str": "success", "completed": True}}

    def download_image(self, reference: ImageReference) -> bytes:
        """Return one valid full-size parity artifact."""

        assert reference.filename == "output.png"
        return _image_bytes()


class _FakeManagedServer:
    """Expose one stopped isolated parity process."""

    def __init__(self, **values: object) -> None:
        """Require both backend owners in the isolated launch."""

        assert values["launch_arguments"] == (
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes",
            "SimpleSyrup",
            "substitute-backend",
            "comfyui-ppm",
        )
        self.running = SimpleNamespace(
            system_stats={"device": "test"},
            client=_FakeClient(),
            port=54321,
            process=SimpleNamespace(is_running=False),
        )

    def __enter__(self) -> object:
        """Return the ready fake process."""

        return self.running

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Leave the fake process stopped."""

        del exc_type, exc, traceback


def test_global_style_parity_preserves_order_and_cleans_links(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Finalize reference then candidate only after exact cleanup."""

    _FakeClient.class_sets = []
    inventory = visual_inventory(tmp_path)
    (tmp_path / "input").mkdir()
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

    result = runner.execute_parity_comparison(
        artifacts,
        inventory=inventory,
        comfy_root=tmp_path,
        readiness_timeout=10.0,
        prompt_timeout=30.0,
        global_style=True,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["output_order"] == ["reference", "candidate"]
    assert payload["controls"]["lora_count"] == 1
    assert "AttentionCouplePPM" in _FakeClient.class_sets[0]
    assert "SimpleSyrup.KSamplerAttentionCoupling" in _FakeClient.class_sets[1]
    assert not (model_root / "checkpoints" / "simple_syrup_u11").exists()
    assert not (model_root / "loras" / "simple_syrup_u11").exists()


def _image_bytes() -> bytes:
    """Return one non-constant anonymous 1024-square PNG."""

    image = Image.new("RGB", (1024, 1024), (10, 20, 30))
    image.putpixel((0, 0), (255, 255, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
