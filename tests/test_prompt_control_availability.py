# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Prompt Control availability detection."""

from __future__ import annotations

import sys
from importlib import invalidate_caches
from pathlib import Path

import pytest

from simple_syrup.runtime.prompt_control_availability import (
    find_prompt_control_install,
    prompt_control_is_available,
)


def test_find_prompt_control_install_detects_sibling_checkout(
    tmp_path: Path,
) -> None:
    """A sibling Prompt Control checkout with nodes_lazy.py is available."""

    package_path = tmp_path / "comfyui-prompt-control" / "prompt_control"
    package_path.mkdir(parents=True)
    (package_path / "nodes_lazy.py").write_text("", encoding="utf-8")

    availability = find_prompt_control_install(custom_nodes_root=tmp_path)

    assert availability.is_available is True
    assert availability.root_path == tmp_path / "comfyui-prompt-control"


def test_find_prompt_control_install_reports_missing_sibling(
    tmp_path: Path,
) -> None:
    """Missing Prompt Control code is reported as unavailable."""

    availability = find_prompt_control_install(custom_nodes_root=tmp_path)

    assert availability.is_available is False
    assert availability.root_path is None


def test_find_prompt_control_install_does_not_import_lazy_nodes(
    tmp_path: Path,
) -> None:
    """Availability checks inspect files without importing nodes_lazy."""

    package_path = tmp_path / "comfyui-prompt-control" / "prompt_control"
    package_path.mkdir(parents=True)
    (package_path / "nodes_lazy.py").write_text("", encoding="utf-8")
    sys.modules.pop("prompt_control.nodes_lazy", None)

    availability = find_prompt_control_install(custom_nodes_root=tmp_path)

    assert availability.is_available is True
    assert "prompt_control.nodes_lazy" not in sys.modules


def test_find_prompt_control_install_detects_python_path_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt Control can be detected when it is already on Python path."""

    package_path = tmp_path / "prompt_control"
    package_path.mkdir()
    (package_path / "nodes_lazy.py").write_text("", encoding="utf-8")
    sys.modules.pop("prompt_control", None)
    sys.modules.pop("prompt_control.nodes_lazy", None)
    monkeypatch.syspath_prepend(str(tmp_path))
    invalidate_caches()

    availability = find_prompt_control_install(custom_nodes_root=tmp_path / "missing")

    assert availability.is_available is True
    assert availability.root_path == tmp_path
    assert "prompt_control.nodes_lazy" not in sys.modules


def test_prompt_control_is_available_uses_default_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convenience lookup returns the default resolver state."""

    monkeypatch.setattr(
        "simple_syrup.runtime.prompt_control_availability.find_prompt_control_install",
        lambda: type(
            "Availability",
            (),
            {"is_available": True, "root_path": None},
        )(),
    )

    assert prompt_control_is_available() is True
