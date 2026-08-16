# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose readable labeled review sheets from retained U11 source images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.comfy_api import JsonObject

_TILE_SIZE = 640
_CAPTION_HEIGHT = 82
_COLUMNS = 3


@dataclass(frozen=True, slots=True)
class VisualReviewGroup:
    """Declare one labeled sheet and its ordered artifact identities."""

    sheet_id: str
    title: str
    artifact_ids: tuple[str, ...]


REVIEW_GROUPS = (
    VisualReviewGroup(
        "characters",
        "SDXL regional character ownership",
        (
            "baseline--full",
            "character-left-prompt-control--full",
            "character-left--full",
            "character-right-prompt-control--full",
            "character-right--full",
            "different-characters-prompt-control--full",
            "different-characters-left-adapter-control--full",
            "different-characters--full",
        ),
    ),
    VisualReviewGroup(
        "styles",
        "SDXL global and regional style composition",
        (
            "baseline--full",
            "global-style-control--full",
            "global-style-full-regional-control--full",
            "global-style-layout-first-control--full",
            "global-style-right-character-prompt-control--full",
            "global-style-regional-character--full",
            "regional-style--full",
            "regional-style-right--full",
            "same-style-global-left--full",
            "multiple-left-style-prompt-control--full",
            "multiple-left-style-only-control--full",
            "multiple-left-model-only-style--full",
            "multiple-left-scheduled-style-only--full",
            "multiple-left-midpoint-style-only--full",
            "multiple-left-minimal-window-style-only--full",
            "multiple-left--full",
            "same-style-both--full",
        ),
    ),
    VisualReviewGroup(
        "schedules-masks",
        "SDXL schedules and mask geometry",
        (
            "scheduled-right-character--full",
            "soft-overlap--full",
            "uncovered-center--full",
            "ra06-independent-schedules--full",
            "ra06-soft-overlap--full",
            "ra06-uncovered-center--full",
        ),
    ),
    VisualReviewGroup(
        "spatial-modes",
        "SDXL full and 1.5x refinement geometry",
        (
            "spatial-mode-global-style-regional-character--full",
            "spatial-mode-global-style-regional-character--tiled-1.5x",
            "spatial-mode-global-style-regional-character--contextual-1.5x",
        ),
    ),
)


def build_visual_review_sheets(result_path: Path) -> tuple[Path, ...]:
    """Build all currently populated review groups without altering source pixels."""

    root = result_path.resolve().parent
    payload = _read_result(result_path)
    observations = _observations(payload)
    output_root = root / "review-sheets"
    output_root.mkdir(exist_ok=True)
    created: list[Path] = []
    manifest: list[JsonObject] = []
    for group in REVIEW_GROUPS:
        selected = tuple(
            observations[artifact_id]
            for artifact_id in group.artifact_ids
            if artifact_id in observations
        )
        if not selected:
            continue
        path = output_root / f"{group.sheet_id}.png"
        _render_sheet(path, root=root, title=group.title, observations=selected)
        created.append(path)
        manifest.append(
            {
                "sheet_id": group.sheet_id,
                "title": group.title,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "artifact_ids": [item["artifact_id"] for item in selected],
            }
        )
    _write_json(output_root / "review-manifest.json", {"sheets": manifest})
    return tuple(created)


def _render_sheet(
    path: Path,
    *,
    root: Path,
    title: str,
    observations: tuple[JsonObject, ...],
) -> None:
    """Render one fixed-grid sheet with uncropped contained source images."""

    rows = (len(observations) + _COLUMNS - 1) // _COLUMNS
    title_height = 72
    canvas = Image.new(
        "RGB",
        (_TILE_SIZE * _COLUMNS, title_height + rows * (_TILE_SIZE + _CAPTION_HEIGHT)),
        "#121212",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    caption_font = ImageFont.load_default(size=18)
    draw.text((24, 20), title, fill="white", font=font)
    for index, observation in enumerate(observations):
        column = index % _COLUMNS
        row = index // _COLUMNS
        x = column * _TILE_SIZE
        y = title_height + row * (_TILE_SIZE + _CAPTION_HEIGHT)
        image_path = root / _required_text(observation, "image_file")
        with Image.open(image_path) as source:
            contained = ImageOps.contain(
                source.convert("RGB"),
                (_TILE_SIZE, _TILE_SIZE),
                method=Image.Resampling.LANCZOS,
            )
        image_x = x + (_TILE_SIZE - contained.width) // 2
        image_y = y + (_TILE_SIZE - contained.height) // 2
        canvas.paste(contained, (image_x, image_y))
        label = (
            f"{_required_text(observation, 'artifact_id')}\n"
            f"{_required_text(observation, 'label')}"
        )
        draw.multiline_text(
            (x + 12, y + _TILE_SIZE + 8),
            label,
            fill="white",
            font=caption_font,
            spacing=4,
        )
    temporary = path.with_suffix(".png.tmp")
    canvas.save(temporary, format="PNG")
    temporary.replace(path)


def _read_result(path: Path) -> JsonObject:
    """Decode one retained result object or fail on invalid evidence."""

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("U11 result is invalid JSON.") from error
    if not isinstance(payload, dict):
        raise TypeError("U11 result must be a JSON object.")
    return cast(JsonObject, payload)


def _observations(payload: JsonObject) -> dict[str, JsonObject]:
    """Index retained observations by unique artifact identity."""

    raw = payload.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ValueError("U11 result contains no visual observations.")
    result: dict[str, JsonObject] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("U11 visual observation must be an object.")
        observation = cast(JsonObject, item)
        artifact_id = _required_text(observation, "artifact_id")
        if artifact_id in result:
            raise ValueError(f"Duplicate U11 artifact identity: {artifact_id!r}.")
        result[artifact_id] = observation
    return result


def _required_text(payload: JsonObject, key: str) -> str:
    """Return one required non-empty JSON text value."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"U11 observation omits required text {key!r}.")
    return value


def _write_json(path: Path, payload: JsonObject) -> None:
    """Atomically persist stable review-sheet evidence."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
