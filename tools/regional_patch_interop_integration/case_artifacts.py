"""Persist one validated P9.7 image or rejection artifact."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import cast

from PIL import Image

from tools.comfy_api import ImageReference, JsonObject

from .history import (
    RegionalPatchInteropError,
    RegionalPatchInteropHistory,
    RegionalPatchInteropSuccess,
)
from .matrix import RegionalPatchInteropCase


class RegionalPatchInteropCaseArtifactWriter:
    """Own accepted-image and rejected-sidecar publication for one root."""

    def __init__(self, root: Path) -> None:
        """Retain one existing resolved artifact directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P9.7 case artifact root must already exist.")

    def write(
        self,
        case: RegionalPatchInteropCase,
        observed: RegionalPatchInteropHistory,
        observation: JsonObject,
        *,
        image_bytes: bytes | None,
        source_image_bytes: bytes | None,
    ) -> Path:
        """Publish one outcome and add its exact values to the observation."""

        if isinstance(observed, RegionalPatchInteropSuccess):
            return self._write_success(
                case,
                observed,
                observation,
                image_bytes=image_bytes,
                source_image_bytes=source_image_bytes,
            )
        if isinstance(observed, RegionalPatchInteropError):
            return self._write_rejection(
                case,
                observed,
                observation,
                image_bytes=image_bytes,
                source_image_bytes=source_image_bytes,
            )
        raise TypeError("P9.7 history result has an invalid type.")

    def _write_success(
        self,
        case: RegionalPatchInteropCase,
        observed: RegionalPatchInteropSuccess,
        observation: JsonObject,
        *,
        image_bytes: bytes | None,
        source_image_bytes: bytes | None,
    ) -> Path:
        """Persist one accepted final image and optional 1024 source image."""

        if image_bytes is None:
            raise ValueError("P9.7 accepted case is missing final image bytes.")
        _validate_image(image_bytes, expected_size=case.image_size)
        image_path = self._root / f"{case.case_id}.png"
        image_path.write_bytes(image_bytes)
        observation.update(
            {
                "metrics": observed.metrics,
                "diagnostics": observed.diagnostics,
                "image_file": image_path.name,
                "image_sha256": _sha256(image_bytes),
                "image_size_bytes": len(image_bytes),
                "image_reference": _image_reference(observed.image_reference),
                "exception_type": None,
                "exception_message": None,
            }
        )
        if observed.source_image_reference is None:
            if source_image_bytes is not None:
                raise ValueError("P9.7 full case supplied an unexpected source image.")
            observation["source_image"] = None
        else:
            if source_image_bytes is None:
                raise ValueError("P9.7 spatial case is missing its source image.")
            _validate_image(source_image_bytes, expected_size=(1024, 1024))
            source_path = self._root / f"{case.case_id}-source.png"
            source_path.write_bytes(source_image_bytes)
            observation["source_image"] = {
                "file": source_path.name,
                "sha256": _sha256(source_image_bytes),
                "size_bytes": len(source_image_bytes),
                "reference": _image_reference(observed.source_image_reference),
            }
        return image_path

    def _write_rejection(
        self,
        case: RegionalPatchInteropCase,
        observed: RegionalPatchInteropError,
        observation: JsonObject,
        *,
        image_bytes: bytes | None,
        source_image_bytes: bytes | None,
    ) -> Path:
        """Persist one named conflict with explicit zero-output evidence."""

        if image_bytes is not None or source_image_bytes is not None:
            raise ValueError("P9.7 rejected cases cannot publish image bytes.")
        rejection_path = self._root / f"{case.case_id}.rejection.json"
        payload = {
            "case_id": case.case_id,
            "label": case.label,
            "status": "rejected_before_sampling",
            "model_call_count": 0,
            "diagnostic_record_count": 0,
            "image": None,
            "exception_type": observed.exception_type,
            "exception_message": observed.exception_message,
            "executed_node_ids": list(observed.executed_node_ids),
        }
        _write_json(rejection_path, payload)
        observation.update(
            {
                "metrics": None,
                "diagnostics": None,
                "image_file": None,
                "source_image": None,
                "exception_type": observed.exception_type,
                "exception_message": observed.exception_message,
                "rejection_file": rejection_path.name,
                "rejection_sha256": _sha256(rejection_path.read_bytes()),
            }
        )
        return rejection_path


def _validate_image(value: bytes, *, expected_size: tuple[int, int]) -> None:
    """Require one non-flat PNG at the exact declared dimensions."""

    if not value:
        raise ValueError("P9.7 image bytes must not be empty.")
    with Image.open(BytesIO(value)) as image:
        if image.format != "PNG" or image.size != expected_size:
            raise ValueError(f"P9.7 image must be a {expected_size} PNG.")
        extrema = cast(
            tuple[tuple[int, int], ...],
            image.convert("RGB").getextrema(),
        )
    if not any(high > low for low, high in extrema):
        raise ValueError("P9.7 image must contain non-flat visual output.")


def _image_reference(reference: ImageReference) -> JsonObject:
    """Return one serializable Comfy image reference."""

    return {
        "filename": reference.filename,
        "subfolder": reference.subfolder,
        "type": reference.output_type,
    }


def _write_json(path: Path, payload: object) -> None:
    """Write one stable UTF-8 JSON sidecar."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(value: bytes) -> str:
    """Hash one complete persisted artifact value."""

    return hashlib.sha256(value).hexdigest()
