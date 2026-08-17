# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preserve one native global-LoRA original and its visible condition label."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.comfy_api import JsonObject
from tools.comfy_integration.portable_font import load_label_font

from .workflow import GLOBAL_LORA_STRENGTH, BuiltNativeGlobalLoraWorkflow


class NativeGlobalLoraResultRecorder:
    """Own exact and labeled evidence for one individually presented image."""

    def __init__(self, root: Path) -> None:
        """Retain one existing managed artifact root."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("Native global-LoRA artifact root must exist.")
        self._observation: JsonObject | None = None

    def record(
        self,
        workflow: BuiltNativeGlobalLoraWorkflow,
        *,
        history: JsonObject,
        prompt_id: str,
        image_bytes: bytes,
        wall_runtime_ms: float,
    ) -> tuple[Path, Path]:
        """Persist one exact original plus one full-resolution labeled view."""

        if self._observation is not None:
            raise RuntimeError("Native global-LoRA result is already recorded.")
        _require_success(history)
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            image = decoded.convert("RGB")
        if image.size != (1024, 1024):
            raise ValueError("Native global-LoRA output must be 1024 by 1024.")
        case_root = self._root / workflow.case.value
        case_root.mkdir(parents=False, exist_ok=False)
        original = case_root / "original.png"
        original.write_bytes(image_bytes)
        labeled = case_root / "labeled.png"
        _write_labeled(image, labeled, workflow.case.label)
        _write_json(case_root / "workflow.json", workflow.prompt)
        _write_json(case_root / "history.json", history)
        self._observation = {
            "case_id": workflow.case.value,
            "label": workflow.case.label,
            "prompt_id": prompt_id,
            "applies_lora": workflow.case.applies_lora,
            "model_strength": (
                GLOBAL_LORA_STRENGTH if workflow.case.applies_lora else 0.0
            ),
            "clip_strength": (
                GLOBAL_LORA_STRENGTH if workflow.case.applies_lora else 0.0
            ),
            "wall_runtime_ms": wall_runtime_ms,
            "original_file": str(original.relative_to(self._root)),
            "original_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "labeled_file": str(labeled.relative_to(self._root)),
            "labeled_sha256": _file_sha256(labeled),
        }
        return original, labeled

    def finalize(
        self,
        *,
        system_stats: JsonObject,
        server_cleanup: bool,
        model_cleanup: bool,
    ) -> Path:
        """Require complete cleanup and write one authoritative result record."""

        if self._observation is None:
            raise RuntimeError("Native global-LoRA output was not recorded.")
        if not server_cleanup or not model_cleanup:
            raise RuntimeError("Native global-LoRA managed cleanup is incomplete.")
        result = self._root / "native-global-lora-result.json"
        _write_json(
            result,
            {
                "schema_version": 1,
                "status": "completed",
                "observation": self._observation,
                "system_stats": system_stats,
                "cleanup": {"server": True, "model_links": True},
            },
        )
        return result


def _write_labeled(image: Image.Image, path: Path, label: str) -> None:
    """Add an in-image condition header without resizing the source output."""

    header_height = 112
    canvas = Image.new("RGB", (1024, 1024 + header_height), (20, 20, 20))
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = load_label_font(30)
    detail_font = load_label_font(21)
    draw.text((24, 18), label, fill="white", font=title_font)
    draw.text(
        (24, 66),
        "Locked native SDXL prompt · seed · sampler · 1024 output",
        fill=(200, 200, 200),
        font=detail_font,
    )
    canvas.save(path, format="PNG", compress_level=9)


def _require_success(history: JsonObject) -> None:
    """Require one successful Comfy terminal history."""

    status = history.get("status")
    if not isinstance(status, dict) or status.get("status_str") != "success":
        raise ValueError("Native global-LoRA history must report success.")


def _file_sha256(path: Path) -> str:
    """Return the exact file hash for one labeled artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write deterministic human-readable JSON evidence."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
