# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist and label the bounded standard-UNet parity evidence."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageFont

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.matrix import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
)
from tools.sdxl_attention_coupling_integration.sampling_controls import (
    SDXL_VISUAL_SAMPLING,
)

from .global_style import ParityGlobalStyle
from .ownership import REGIONAL_OWNERSHIP_STRENGTH
from .workflow import BuiltParityWorkflow, ParityBackend

_EXPECTED_ORDER = (ParityBackend.REFERENCE, ParityBackend.CANDIDATE)


class ParityResultRecorder:
    """Own incremental evidence persistence and the labeled review artifact."""

    def __init__(
        self,
        root: Path,
        *,
        backends: tuple[ParityBackend, ...] = _EXPECTED_ORDER,
        global_style: ParityGlobalStyle | None = None,
    ) -> None:
        """Require one existing managed-run root and initialize the result."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise FileNotFoundError("Parity result root does not exist.")
        if (
            not isinstance(backends, tuple)
            or not backends
            or any(not isinstance(backend, ParityBackend) for backend in backends)
            or tuple(backend for backend in _EXPECTED_ORDER if backend in backends)
            != backends
        ):
            raise ValueError("Parity result backends must use canonical order.")
        if global_style is not None and not isinstance(global_style, ParityGlobalStyle):
            raise TypeError("Parity result global style has an invalid type.")
        self._global_style = global_style
        self._backends = backends
        self._observations: list[JsonObject] = []
        self._persist("starting")

    def record(
        self,
        workflow: BuiltParityWorkflow,
        *,
        history: JsonObject,
        prompt_id: str,
        image_bytes: bytes,
        wall_runtime_ms: float,
    ) -> Path:
        """Persist one unmodified source image and its complete execution evidence."""

        expected = self._backends[len(self._observations)]
        if workflow.backend is not expected:
            raise ValueError(
                "Parity outputs must be recorded reference then candidate."
            )
        if wall_runtime_ms <= 0:
            raise ValueError("Parity runtime must be positive.")
        dimensions, dynamic_range = _image_evidence(image_bytes)
        if dimensions != (SOURCE_WIDTH, SOURCE_HEIGHT) or dynamic_range == 0:
            raise ValueError(
                "Parity output must be one non-constant source-size image."
            )
        backend_root = self._root / workflow.backend.value
        backend_root.mkdir(exist_ok=False)
        image_path = backend_root / "full.png"
        image_path.write_bytes(image_bytes)
        _write_json(backend_root / "workflow.json", cast(JsonObject, workflow.prompt))
        _write_json(backend_root / "history.json", history)
        self._observations.append(
            {
                "backend": workflow.backend.value,
                "label": _label(workflow.backend),
                "prompt_id": prompt_id,
                "image_file": str(image_path.relative_to(self._root)),
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_size_bytes": len(image_bytes),
                "image_width": dimensions[0],
                "image_height": dimensions[1],
                "image_dynamic_range": dynamic_range,
                "wall_runtime_ms": wall_runtime_ms,
            }
        )
        self._persist("running")
        return image_path

    def finalize(
        self,
        *,
        system_stats: JsonObject,
        server_cleanup: bool,
        model_cleanup: bool,
        mask_cleanup: bool,
        mask_evidence: tuple[dict[str, object], ...],
    ) -> Path:
        """Create the review sheet only after both outputs and cleanup are proven."""

        observed = tuple(item["backend"] for item in self._observations)
        if observed != tuple(backend.value for backend in self._backends):
            raise ValueError("Parity comparison is incomplete.")
        if not all((server_cleanup, model_cleanup, mask_cleanup)):
            raise ValueError("Parity comparison requires exact external cleanup.")
        review_path = self._root / "review.png"
        _build_review(review_path, root=self._root, observations=self._observations)
        return self._persist(
            "completed",
            extra={
                "controls": {
                    "seed": SDXL_VISUAL_SAMPLING.seed,
                    "steps": SDXL_VISUAL_SAMPLING.steps,
                    "cfg": SDXL_VISUAL_SAMPLING.cfg,
                    "sampler": SDXL_VISUAL_SAMPLING.sampler,
                    "scheduler": SDXL_VISUAL_SAMPLING.scheduler,
                    "width": SOURCE_WIDTH,
                    "height": SOURCE_HEIGHT,
                    "regional_weight": REGIONAL_OWNERSHIP_STRENGTH,
                    "region_mask_feather": 0,
                    "lora_count": 1 if self._global_style is not None else 0,
                    "global_lora_strength": (
                        self._global_style.strength
                        if self._global_style is not None
                        else 0.0
                    ),
                    "global_style_trigger_scope": (
                        "base_positive_only"
                        if self._global_style is not None
                        else "absent"
                    ),
                },
                "system_stats": system_stats,
                "mask_evidence": list(mask_evidence),
                "review_file": review_path.name,
                "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                "cleanup": {"server": True, "model_link": True, "masks": True},
            },
        )

    def _persist(self, status: str, *, extra: JsonObject | None = None) -> Path:
        """Atomically replace the authoritative parity result object."""

        payload: JsonObject = {
            "status": status,
            "output_order": [backend.value for backend in self._backends],
            "observations": self._observations,
        }
        if extra is not None:
            payload.update(extra)
        path = self._root / "parity-result.json"
        _write_json(path, payload)
        return path


def _image_evidence(image_bytes: bytes) -> tuple[tuple[int, int], int]:
    """Return decoded dimensions and observed RGB dynamic range."""

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        extrema = cast(
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]], rgb.getextrema()
        )
        dynamic_range = max(high - low for low, high in extrema)
        return rgb.size, dynamic_range


def _build_review(
    path: Path,
    *,
    root: Path,
    observations: list[JsonObject],
) -> None:
    """Render two uncropped source images with explicit backend labels."""

    caption_height = 62
    canvas = Image.new(
        "RGB",
        (SOURCE_WIDTH * len(observations), SOURCE_HEIGHT + caption_height),
        "#121212",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    for index, observation in enumerate(observations):
        image_file = observation.get("image_file")
        label = observation.get("label")
        if not isinstance(image_file, str) or not isinstance(label, str):
            raise ValueError("Parity observation is missing review metadata.")
        with Image.open(root / image_file) as source:
            canvas.paste(source.convert("RGB"), (index * SOURCE_WIDTH, caption_height))
        draw.text((index * SOURCE_WIDTH + 20, 18), label, fill="white", font=font)
    temporary = path.with_suffix(".png.tmp")
    canvas.save(temporary, format="PNG")
    temporary.replace(path)


def _label(backend: ParityBackend) -> str:
    """Return one unambiguous user-facing artifact label."""

    if backend is ParityBackend.REFERENCE:
        return "REFERENCE - installed Attention Couple"
    return "CANDIDATE - SimpleSyrup standard-UNet backend"


def _write_json(path: Path, payload: JsonObject) -> None:
    """Atomically persist stable JSON evidence."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
