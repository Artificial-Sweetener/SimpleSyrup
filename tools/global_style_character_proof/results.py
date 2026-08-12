"""Persist and label global-style plus regional-character visual evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.comfy_api import ImageReference, JsonObject

from .matrix import GlobalStyleCharacterCase


class GlobalStyleCharacterProofRecorder:
    """Own ordered images, sidecars, and the full comparison sheet."""

    def __init__(self, root: Path) -> None:
        """Retain one managed artifact directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("Global style/character proof root must exist.")
        self._observations: list[JsonObject] = []
        self._images: dict[str, Path] = {}

    def record(
        self,
        case: GlobalStyleCharacterCase,
        *,
        workflow: dict[str, JsonObject],
        history: JsonObject,
        prompt_id: str,
        reference: ImageReference,
        image_bytes: bytes,
    ) -> None:
        """Require success and persist one full-resolution case."""

        status = history.get("status")
        if not isinstance(status, dict) or status.get("status_str") != "success":
            raise ValueError("Global style/character proof case must succeed.")
        path = self._root / f"{case.case_id}.png"
        path.write_bytes(image_bytes)
        self._images[case.case_id] = path
        _write_json(self._root / f"{case.case_id}.workflow.json", workflow)
        _write_json(self._root / f"{case.case_id}.history.json", history)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "status": "success",
                "prompt_id": prompt_id,
                "global_style_strength": case.global_style_strength,
                "regional_character_strength": 1.0,
                "image_file": path.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_reference": {
                    "filename": reference.filename,
                    "subfolder": reference.subfolder,
                    "type": reference.output_type,
                },
            }
        )

    def finalize(
        self,
        definitions: tuple[GlobalStyleCharacterCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
        masks_removed: bool,
    ) -> Path:
        """Write the labeled comparison after complete cleanup proof."""

        expected = tuple(case.case_id for case in definitions)
        observed = tuple(str(item["case_id"]) for item in self._observations)
        if observed != expected:
            raise ValueError("Global style/character observations are incomplete.")
        if not cleanup_verified or not masks_removed:
            raise ValueError("Global style/character cleanup is incomplete.")
        comparison = self._comparison(definitions)
        result = self._root / "global-style-character-proof.json"
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

    def _comparison(
        self,
        definitions: tuple[GlobalStyleCharacterCase, ...],
    ) -> Path:
        """Place all three complete images side by side with exact labels."""

        header = 96
        panel = 1024
        canvas = Image.new("RGB", (panel * 3, panel + header), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(r"<SYSTEM_FONT>", 25)
        for index, case in enumerate(definitions):
            with Image.open(self._images[case.case_id]) as source:
                image = source.convert("RGB")
            if image.size != (panel, panel):
                raise ValueError("Global style/character images must be 1024 square.")
            left = index * panel
            canvas.paste(image, (left, header))
            draw.text((left + 18, 32), case.label, fill="white", font=font)
        path = self._root / "global-style-character_a-region__labeled-comparison.png"
        canvas.save(path, format="PNG")
        return path


def _write_json(path: Path, value: object) -> None:
    """Write deterministic human-readable JSON evidence."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
