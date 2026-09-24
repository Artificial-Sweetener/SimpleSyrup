# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load Ultralytics models behind a narrow runtime adapter."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class UltralyticsDetectorModel:
    """Store a loaded Ultralytics detector with SimpleSyrup metadata."""

    model_name: str
    model_path: Path
    model: Any
    task: str
    names: dict[int, str]
    supports_segmentation: bool


class UltralyticsModelAdapter:
    """Construct detector models through the optional Ultralytics runtime."""

    def __init__(self, ultralytics_module: ModuleType | None = None) -> None:
        """Create the adapter with an optional runtime module override."""

        self._ultralytics_module = ultralytics_module

    def load(self, model_name: str, model_path: Path) -> UltralyticsDetectorModel:
        """Load one detector checkpoint and expose normalized metadata."""

        model_class = getattr(self._ultralytics(), "YOLO", None)
        if model_class is None:
            raise RuntimeError(
                "Ultralytics support requires a module exposing the YOLO class."
            )

        try:
            raw_model = model_class(str(model_path))
        except Exception as exc:
            LOGGER.error(
                "Failed to load Ultralytics model",
                extra={
                    "operation": "load_ultralytics_model",
                    "model_name": model_name,
                    "model_path": str(model_path),
                },
                exc_info=True,
            )
            raise RuntimeError(
                f"Ultralytics model '{model_name}' could not be loaded from "
                f"'{model_path}'."
            ) from exc

        task = _model_task(model_name, raw_model)
        detector_model = UltralyticsDetectorModel(
            model_name=model_name,
            model_path=model_path,
            model=raw_model,
            task=task,
            names=_model_names(raw_model),
            supports_segmentation=task in {"segment", "segm"},
        )
        LOGGER.info(
            "Ultralytics model loaded",
            extra={
                "operation": "load_ultralytics_model",
                "model_name": model_name,
                "model_path": str(model_path),
                "task": task,
                "device": _model_device_hint(raw_model),
            },
        )
        return detector_model

    def _ultralytics(self) -> ModuleType:
        """Import Ultralytics lazily and fail with an actionable message."""

        if self._ultralytics_module is not None:
            return self._ultralytics_module
        try:
            module = importlib.import_module("ultralytics")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ultralytics support requires the 'ultralytics' package in the "
                "ComfyUI virtual environment."
            ) from exc
        if not isinstance(module, ModuleType):
            raise TypeError("ultralytics import did not return a module.")
        self._ultralytics_module = module
        return module


def _model_task(model_name: str, raw_model: object) -> str:
    """Infer detector task from choice prefix or model metadata."""

    normalized_name = model_name.replace("\\", "/")
    if normalized_name.startswith("segm/"):
        return "segment"
    if normalized_name.startswith("bbox/"):
        return "detect"

    task = getattr(raw_model, "task", None)
    if isinstance(task, str) and task:
        return task
    return "detect"


def _model_names(raw_model: object) -> dict[int, str]:
    """Extract class names from a loaded Ultralytics model."""

    names = getattr(raw_model, "names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _model_device_hint(raw_model: object) -> str:
    """Return a best-effort Ultralytics device hint for diagnostics."""

    direct_device = getattr(raw_model, "device", None)
    if direct_device is not None:
        return str(direct_device)
    inner_model = getattr(raw_model, "model", None)
    inner_device = getattr(inner_model, "device", None)
    if inner_device is not None:
        return str(inner_device)
    return "runtime-owned"
