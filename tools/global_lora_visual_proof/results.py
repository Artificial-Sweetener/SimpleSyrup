# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate and label global/regional PRIMARY_ADAPTER visual proof artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.comfy_api import ImageReference, JsonObject

from .matrix import GlobalLoraVisualCase

_OVERLAP_MESSAGE = "Regional Anima LoRA content is already applied globally"


class GlobalLoraVisualProofRecorder:
    """Own ordered proof observations, full images, and the comparison sheet."""

    def __init__(self, root: Path) -> None:
        """Retain one managed artifact directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("Global LoRA proof root must already exist.")
        self._observations: list[JsonObject] = []
        self._images: dict[str, Path] = {}

    def record_success(
        self,
        case: GlobalLoraVisualCase,
        *,
        workflow: dict[str, JsonObject],
        history: JsonObject,
        prompt_id: str,
        reference: ImageReference,
        image_bytes: bytes,
    ) -> Path:
        """Persist one successful full-resolution image and exact sidecars."""

        if case.expect_overlap_rejection:
            raise ValueError("Duplicate LoRA case cannot be recorded as success.")
        _require_status(history, "success")
        path = self._root / f"{case.case_id}.png"
        path.write_bytes(image_bytes)
        self._images[case.case_id] = path
        self._write_sidecars(case, workflow, history)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "status": "success",
                "prompt_id": prompt_id,
                "image_file": path.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_reference": {
                    "filename": reference.filename,
                    "subfolder": reference.subfolder,
                    "type": reference.output_type,
                },
            }
        )
        return path

    def record_overlap_rejection(
        self,
        case: GlobalLoraVisualCase,
        *,
        workflow: dict[str, JsonObject],
        history: JsonObject,
        prompt_id: str,
    ) -> None:
        """Persist the intentional pre-sampling duplicate rejection."""

        if not case.expect_overlap_rejection:
            raise ValueError("Success case cannot be recorded as a rejection.")
        _require_status(history, "error")
        serialized = json.dumps(history, sort_keys=True)
        if (
            _OVERLAP_MESSAGE not in serialized
            or "adapter-a.safetensors" not in serialized
        ):
            raise ValueError("Duplicate case lacks the exact overlap diagnostic.")
        self._write_sidecars(case, workflow, history)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "status": "rejected_before_sampling",
                "prompt_id": prompt_id,
                "diagnostic": _OVERLAP_MESSAGE,
            }
        )

    def finalize(
        self,
        definitions: tuple[GlobalLoraVisualCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
        masks_removed: bool,
    ) -> Path:
        """Require complete ordered evidence and write its labeled comparison."""

        expected = tuple(case.case_id for case in definitions)
        observed = tuple(str(item["case_id"]) for item in self._observations)
        if observed != expected:
            raise ValueError("Global LoRA proof observations are incomplete.")
        if not cleanup_verified or not masks_removed:
            raise ValueError("Global LoRA proof cleanup is incomplete.")
        comparison = self._write_comparison(definitions)
        result = self._root / "global-lora-visual-proof.json"
        _write_json(
            result,
            {
                "schema_version": 1,
                "status": "completed",
                "observations": self._observations,
                "comparison_file": comparison.name,
                "system_stats": system_stats,
                "cleanup": {
                    "process_stopped": True,
                    "port_available": True,
                    "masks_removed": True,
                },
            },
        )
        return result

    def _write_comparison(
        self,
        definitions: tuple[GlobalLoraVisualCase, ...],
    ) -> Path:
        """Render four outputs and the duplicate diagnostic in a labeled grid."""

        panel = 768
        header = 80
        canvas = Image.new("RGB", (panel * 3, (panel + header) * 2), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        title_font = ImageFont.truetype(r"<SYSTEM_FONT>", 24)
        body_font = ImageFont.truetype(r"<SYSTEM_FONT>", 20)
        for index, case in enumerate(definitions):
            column = index % 3
            row = index // 3
            left = column * panel
            top = row * (panel + header)
            draw.text((left + 18, top + 25), case.label, fill="white", font=title_font)
            body_top = top + header
            path = self._images.get(case.case_id)
            if path is not None:
                with Image.open(path) as source:
                    image = source.convert("RGB").resize(
                        (panel, panel), Image.Resampling.LANCZOS
                    )
                canvas.paste(image, (left, body_top))
                continue
            draw.rectangle(
                (left, body_top, left + panel - 1, body_top + panel - 1),
                fill=(45, 28, 28),
                outline=(220, 90, 90),
                width=3,
            )
            lines = (
                "REJECTED BEFORE SAMPLING",
                "The same PRIMARY_ADAPTER weights were already",
                "installed globally on the input model.",
                "Current policy prevents double application.",
            )
            for line_index, line in enumerate(lines):
                draw.text(
                    (left + 48, body_top + 230 + line_index * 42),
                    line,
                    fill=(255, 220, 220),
                    font=body_font,
                )
        path = self._root / "global-lora-placement__labeled-comparison.png"
        canvas.save(path, format="PNG")
        return path

    def _write_sidecars(
        self,
        case: GlobalLoraVisualCase,
        workflow: dict[str, JsonObject],
        history: JsonObject,
    ) -> None:
        """Persist exact submitted workflow and returned history."""

        _write_json(self._root / f"{case.case_id}.workflow.json", workflow)
        _write_json(self._root / f"{case.case_id}.history.json", history)


def _require_status(history: JsonObject, expected: str) -> None:
    """Require one exact Comfy terminal status."""

    status = history.get("status")
    if not isinstance(status, dict) or status.get("status_str") != expected:
        raise ValueError(f"Global LoRA proof history must have status {expected!r}.")


def _write_json(path: Path, value: object) -> None:
    """Write deterministic human-readable JSON."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
