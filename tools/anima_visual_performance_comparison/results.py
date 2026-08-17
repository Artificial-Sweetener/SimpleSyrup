# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist original and visibly labeled Anima comparison artifacts."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.comfy_api import JsonObject
from tools.comfy_integration.portable_font import load_label_font

from .cases import AnimaVisualCase


class AnimaVisualResultRecorder:
    """Own ordered images, timing evidence, and visible labels."""

    def __init__(self, root: Path) -> None:
        """Retain one existing managed artifact directory."""

        self._root = root.resolve()
        self._observations: list[JsonObject] = []

    def record(
        self,
        case: AnimaVisualCase,
        *,
        workflow: dict[str, JsonObject],
        history: JsonObject,
        image_bytes: bytes,
        metrics: JsonObject,
        wall_runtime_ms: float,
    ) -> Path:
        """Persist one original and full-resolution labeled comparison image."""

        case_root = self._root / case.mode.value
        case_root.mkdir(exist_ok=False)
        original = case_root / "original.png"
        original.write_bytes(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
        if image.size != (1024, 1024):
            raise ValueError("Anima comparison image must be 1024-square.")
        labeled = case_root / "labeled.png"
        _label(image, labeled, case.label, metrics, wall_runtime_ms)
        _write_json(case_root / "workflow.json", workflow)
        _write_json(case_root / "history.json", history)
        self._observations.append(
            {
                "case_id": case.mode.value,
                "label": case.label,
                "wall_runtime_ms": wall_runtime_ms,
                "metrics": metrics,
                "original_file": str(original.relative_to(self._root)),
                "labeled_file": str(labeled.relative_to(self._root)),
                "original_sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )
        self._persist("running")
        return labeled

    def finalize(self, cases: tuple[AnimaVisualCase, ...]) -> Path:
        """Complete only after every predeclared case is recorded in order."""

        expected = tuple(case.mode.value for case in cases)
        observed = tuple(str(item["case_id"]) for item in self._observations)
        if observed != expected:
            raise ValueError("Anima visual observations are incomplete.")
        return self._persist("completed")

    def _persist(self, status: str) -> Path:
        """Atomically replace the current result record."""

        path = self._root / "anima-visual-performance-result.json"
        _write_json(
            path,
            {"status": status, "observations": self._observations},
        )
        return path


def _label(
    image: Image.Image,
    path: Path,
    label: str,
    metrics: JsonObject,
    wall_runtime_ms: float,
) -> None:
    """Add an unambiguous timing and topology header above the original."""

    model_runtime = metrics.get("runtime_ms")
    if not isinstance(model_runtime, int | float):
        raise ValueError("Anima comparison metrics are missing runtime_ms.")
    canvas = Image.new("RGB", (1024, 1144), (20, 20, 20))
    canvas.paste(image, (0, 120))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 15), label, fill="white", font=load_label_font(28))
    detail = (
        f"30 steps · 1024×1024 · model {model_runtime / 1000:.2f}s · "
        f"wall {wall_runtime_ms / 1000:.2f}s"
    )
    draw.text(
        (22, 66),
        detail,
        fill=(205, 205, 205),
        font=load_label_font(21),
    )
    canvas.save(path, format="PNG")


def _write_json(path: Path, payload: object) -> None:
    """Atomically write one stable JSON artifact."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
