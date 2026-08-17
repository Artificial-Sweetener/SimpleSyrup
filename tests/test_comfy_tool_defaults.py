# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify portable managed-Comfy defaults and generic fixture selections."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.comfy_integration.anima_fixture_selections import ANIMA_BASE_SELECTIONS
from tools.comfy_integration.default_paths import (
    COMFY_ROOT_ENVIRONMENT_VARIABLE,
    default_benchmark_artifact_root,
    default_comfy_root,
    default_custom_node_root,
    default_input_root,
)
from tools.comfy_integration.portable_font import load_label_font


def test_explicit_comfy_root_owns_every_derived_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Derive all tool paths from one explicit host boundary."""

    monkeypatch.setenv(COMFY_ROOT_ENVIRONMENT_VARIABLE, str(tmp_path))

    assert default_comfy_root() == tmp_path.resolve()
    assert default_input_root() == tmp_path.resolve() / "input"
    assert default_benchmark_artifact_root("family/case") == (
        tmp_path.resolve() / "benchmark_artifacts" / "family" / "case"
    )
    assert default_custom_node_root("sibling/node") == (
        tmp_path.resolve() / "custom_nodes" / "sibling" / "node"
    )


def test_comfy_root_and_suffixes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject relative host roots and attempts to leave managed directories."""

    monkeypatch.setenv(COMFY_ROOT_ENVIRONMENT_VARIABLE, "relative-root")
    with pytest.raises(ValueError, match="must be an absolute path"):
        default_comfy_root()

    monkeypatch.setenv(COMFY_ROOT_ENVIRONMENT_VARIABLE, str(Path.cwd()))
    with pytest.raises(ValueError, match="must stay relative"):
        default_benchmark_artifact_root("../outside")


def test_anima_fixture_selections_are_generic_managed_aliases() -> None:
    """Keep real artifact identities outside repository workflow declarations."""

    assert tuple(category for category, _ in ANIMA_BASE_SELECTIONS) == (
        "diffusion_models",
        "text_encoders",
        "vae",
    )
    assert all(
        selection.startswith("simple_syrup_anima\\")
        for _, selection in ANIMA_BASE_SELECTIONS
    )
    assert all(
        "qwen" not in selection.casefold() for _, selection in ANIMA_BASE_SELECTIONS
    )


def test_label_font_has_no_operating_system_font_dependency() -> None:
    """Load a readable bundled font without consulting a machine font path."""

    assert load_label_font(20).getbbox("SimpleSyrup") is not None
