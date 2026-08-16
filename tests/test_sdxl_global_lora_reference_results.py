# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact and visibly labeled native global-LoRA artifacts."""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from tools.sdxl_global_lora_reference.cases import NativeGlobalLoraCase
from tools.sdxl_global_lora_reference.results import NativeGlobalLoraResultRecorder
from tools.sdxl_global_lora_reference.workflow import BuiltNativeGlobalLoraWorkflow


def test_recorder_preserves_original_and_full_resolution_labeled_view(
    tmp_path: Path,
) -> None:
    """Keep output bytes exact and add a visible header without resampling it."""

    recorder = NativeGlobalLoraResultRecorder(tmp_path)
    source = _image_bytes()
    workflow = BuiltNativeGlobalLoraWorkflow(
        NativeGlobalLoraCase.TRIGGER_CONTROL,
        {"1": {"class_type": "SaveImage", "inputs": {}}},
        "1",
    )
    original, labeled = recorder.record(
        workflow,
        history={"status": {"status_str": "success", "completed": True}},
        prompt_id="prompt",
        image_bytes=source,
        wall_runtime_ms=2.5,
    )
    result = recorder.finalize(
        system_stats={"device": "test"},
        server_cleanup=True,
        model_cleanup=True,
    )

    assert original.read_bytes() == source
    with Image.open(labeled) as image:
        assert image.size == (1024, 1136)
        assert (
            image.crop((0, 112, 1024, 1136)).tobytes()
            == Image.open(io.BytesIO(source)).convert("RGB").tobytes()
        )
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["observation"]["case_id"] == "native-trigger-control"
    assert payload["observation"]["applies_lora"] is False
    assert payload["cleanup"] == {"model_links": True, "server": True}


def _image_bytes() -> bytes:
    """Return one anonymous non-constant 1024-square PNG."""

    image = Image.new("RGB", (1024, 1024), (12, 34, 56))
    image.putpixel((0, 0), (255, 255, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
