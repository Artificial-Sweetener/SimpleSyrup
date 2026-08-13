# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve the model root Comfy will activate during managed startup."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_active_comfy_model_root(comfy_root: Path) -> Path:
    """Return the validated persisted Substitute root or Comfy's default root."""

    root = comfy_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError("Comfy root does not exist.")
    configuration = root / ".substitute" / "model_root.json"
    if not configuration.exists():
        model_root = root / "models"
    else:
        try:
            payload: object = json.loads(configuration.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                "Substitute model-root configuration is invalid JSON."
            ) from error
        if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
            raise ValueError("Substitute model-root configuration schema is invalid.")
        value = payload.get("modelRoot")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Substitute model-root configuration omits modelRoot.")
        model_root = Path(value).expanduser()
        if not model_root.is_absolute():
            raise ValueError("Substitute model root must be absolute.")
    resolved = model_root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError("Active Comfy model root does not exist.")
    return resolved
