# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test repository license header maintenance."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_MODULE = REPO_ROOT / "tools" / "add_license_headers.py"
PROJECT_LINE = "SimpleSyrup - workflow-focused ComfyUI extensions for image generation"
SPDX_LINE = "SPDX-License-Identifier: AGPL-3.0-or-later"


def _load_module(path: Path) -> ModuleType:
    """Load the license header tool directly from its repository path."""

    spec = importlib.util.spec_from_file_location("add_license_headers_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


license_headers = cast(Any, _load_module(TOOLS_MODULE))


def test_copyright_years_stays_single_year_during_start_year() -> None:
    """Keep the initial release year compact while it is still current."""

    assert license_headers._copyright_years(datetime(2026, 5, 21, tzinfo=UTC)) == "2026"


def test_copyright_years_expands_after_start_year() -> None:
    """Render a range when the tool is rerun in a later year."""

    assert (
        license_headers._copyright_years(datetime(2030, 1, 1, tzinfo=UTC))
        == "2026 - 2030"
    )


def test_header_uses_typescript_comment_prefix_for_frontend_source() -> None:
    """Use TypeScript comments for ComfyUI frontend extension files."""

    header = license_headers._header(path=Path("web/src/main.ts"))

    assert header.startswith(f"// {PROJECT_LINE}")
    assert SPDX_LINE in header


def test_direct_vendored_and_generated_source_are_excluded() -> None:
    """Keep generated files and direct upstream vendored runtime outside stamping."""

    should_update = license_headers._should_update

    assert not should_update(
        Path("simple_syrup/third_party/groundingdino_runtime/models/registry.py")
    )
    assert not should_update(
        Path("simple_syrup/third_party/sam_hq_runtime/predictor.py")
    )
    assert not should_update(Path("web/dist/simple-syrup.js"))
    assert should_update(Path("simple_syrup/runtime/a1111_sampling.py"))


def test_tracked_first_party_source_has_project_license_header() -> None:
    """Require tracked SimpleSyrup-owned source to carry the AGPL project header."""

    for source_path in license_headers._tracked_source_files():
        content = (REPO_ROOT / source_path).read_text(encoding="utf-8")
        prefix = license_headers._comment_prefix(source_path)

        assert f"{prefix} {PROJECT_LINE}" in content[:500], source_path
        assert f"{prefix} {SPDX_LINE}" in content[:500], source_path
