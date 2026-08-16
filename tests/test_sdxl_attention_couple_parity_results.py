# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify durable labeled artifacts for the bounded parity comparison."""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from tools.sdxl_attention_couple_parity.global_style import ParityGlobalStyle
from tools.sdxl_attention_couple_parity.results import ParityResultRecorder
from tools.sdxl_attention_couple_parity.workflow import (
    BuiltParityWorkflow,
    ParityBackend,
)


def test_recorder_preserves_source_bytes_and_labels_review(tmp_path: Path) -> None:
    """Keep both originals exact and create one explicit two-column artifact."""

    recorder = ParityResultRecorder(tmp_path)
    source_bytes = _image_bytes()
    for backend in ParityBackend:
        recorder.record(
            BuiltParityWorkflow(
                backend, {"1": {"class_type": "SaveImage", "inputs": {}}}, "1"
            ),
            history={"status": {"status_str": "success", "completed": True}},
            prompt_id=f"prompt-{backend.value}",
            image_bytes=source_bytes,
            wall_runtime_ms=1.0,
        )
        assert (tmp_path / backend.value / "full.png").read_bytes() == source_bytes
    result = recorder.finalize(
        system_stats={},
        server_cleanup=True,
        model_cleanup=True,
        mask_cleanup=True,
        mask_evidence=(),
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["output_order"] == ["reference", "candidate"]
    assert payload["controls"]["regional_weight"] == 0.4
    assert (tmp_path / "review.png").is_file()
    with Image.open(tmp_path / "review.png") as review:
        assert review.size == (2048, 1086)


def test_recorder_declares_global_style_control(tmp_path: Path) -> None:
    """Record the optional generic LoRA control without inventory identity logic."""

    style = ParityGlobalStyle(
        lora_name=r"owned\style.safetensors",
        strength=0.65,
        prompt_g="style G",
        prompt_l="style L",
    )
    recorder = ParityResultRecorder(tmp_path, global_style=style)
    for backend in ParityBackend:
        recorder.record(
            BuiltParityWorkflow(
                backend, {"1": {"class_type": "SaveImage", "inputs": {}}}, "1"
            ),
            history={"status": {"status_str": "success", "completed": True}},
            prompt_id=f"prompt-{backend.value}",
            image_bytes=_image_bytes(),
            wall_runtime_ms=1.0,
        )

    result = recorder.finalize(
        system_stats={},
        server_cleanup=True,
        model_cleanup=True,
        mask_cleanup=True,
        mask_evidence=(),
    )
    controls = json.loads(result.read_text(encoding="utf-8"))["controls"]
    assert controls["lora_count"] == 1
    assert controls["global_lora_strength"] == 0.65
    assert controls["global_style_trigger_scope"] == "base_positive_only"


def test_recorder_supports_one_presented_backend(tmp_path: Path) -> None:
    """Finalize one labeled backend so it can be shown before the next run."""

    recorder = ParityResultRecorder(
        tmp_path,
        backends=(ParityBackend.REFERENCE,),
    )
    recorder.record(
        BuiltParityWorkflow(
            ParityBackend.REFERENCE,
            {"1": {"class_type": "SaveImage", "inputs": {}}},
            "1",
        ),
        history={"status": {"status_str": "success", "completed": True}},
        prompt_id="prompt-reference",
        image_bytes=_image_bytes(),
        wall_runtime_ms=1.0,
    )

    result = recorder.finalize(
        system_stats={},
        server_cleanup=True,
        model_cleanup=True,
        mask_cleanup=True,
        mask_evidence=(),
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["output_order"] == ["reference"]
    with Image.open(tmp_path / "review.png") as review:
        assert review.size == (1024, 1086)


def _image_bytes() -> bytes:
    """Return one non-constant anonymous 1024-square PNG fixture."""

    image = Image.new("RGB", (1024, 1024), "black")
    image.putpixel((0, 0), (255, 255, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
