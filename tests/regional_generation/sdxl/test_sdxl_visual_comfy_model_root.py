# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify managed U11 model links follow Comfy's active prestartup root."""

from __future__ import annotations

from pathlib import Path

from tools.sdxl_attention_coupling_integration.comfy_model_root import (
    resolve_active_comfy_model_root,
)


def test_default_model_root_is_used_without_substitute_config(tmp_path: Path) -> None:
    """Follow normal Comfy startup when no persisted redirect exists."""

    models = tmp_path / "models"
    models.mkdir()

    assert resolve_active_comfy_model_root(tmp_path) == models


def test_absolute_substitute_model_root_overrides_default(tmp_path: Path) -> None:
    """Match the exact prestartup root that the real server will activate."""

    configured = tmp_path / "configured-models"
    configured.mkdir()
    configuration = tmp_path / ".substitute" / "model_root.json"
    configuration.parent.mkdir()
    configuration.write_text(
        '{"schemaVersion":1,"modelRoot":"'
        + str(configured).replace("\\", "\\\\")
        + '"}\n',
        encoding="utf-8",
    )

    assert resolve_active_comfy_model_root(tmp_path) == configured
