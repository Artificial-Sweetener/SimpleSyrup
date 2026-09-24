# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify structural governance and the current architecture inventory."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from support.repository import REPOSITORY_ROOT

from tools.architecture_governance.import_boundaries import validate_import_boundaries
from tools.architecture_governance.metrics import production_line_count
from tools.architecture_governance.soft_reviews import validate_soft_reviews
from tools.architecture_governance.validation import validate_repository


def _write(path: Path, content: str) -> None:
    """Write one isolated governance fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_policy(root: Path) -> None:
    """Create a minimal strict architecture policy."""

    _write(
        root / "governance/architecture/policy.toml",
        """schema_version = 2
[structure]
soft_lines = 2
hard_lines = 4
source_roots = ["product"]
source_files = []
source_extensions = [".py", ".ts"]
excluded_paths = []
[registries]
debt = "governance/architecture/debt.toml"
waivers = "governance/architecture/waivers.toml"
""",
    )
    _write(
        root / "governance/architecture/debt.toml",
        "schema_version = 1\ndebts = []\n",
    )
    _write(
        root / "governance/architecture/waivers.toml",
        "schema_version = 1\nwaivers = []\n",
    )


def test_unclassified_hard_overage_is_blocking(tmp_path: Path) -> None:
    """Require human ownership assessment for every hard-gate file."""

    _write_policy(tmp_path)
    _write(
        tmp_path / "product/mixed.py",
        "\n".join(f"VALUE_{index} = {index}" for index in range(6)),
    )

    diagnostics = validate_repository(tmp_path, today=date(2026, 9, 24))

    assert any(item.rule == "STRUCT003" for item in diagnostics)


def test_typescript_comments_do_not_inflate_production_lines(tmp_path: Path) -> None:
    """Count authored TypeScript while excluding standalone comments."""

    path = tmp_path / "source.ts"
    _write(path, "// heading\n/* block\ncomment */\nconst value = 1;\n")

    assert production_line_count(path) == 1


def test_current_repository_has_no_architecture_governance_errors() -> None:
    """Keep every hard structural finding exactly reviewed."""

    errors = [
        item
        for item in validate_repository(REPOSITORY_ROOT)
        if item.severity == "error"
    ]

    assert errors == []


def test_current_repository_has_no_unreviewed_import_boundary_debt() -> None:
    """Keep every dependency-direction violation exactly inventoried."""

    assert validate_import_boundaries(REPOSITORY_ROOT) == []


def test_current_repository_has_exact_soft_ceiling_reviews() -> None:
    """Keep every advisory structural warning source-reviewed and current."""

    assert validate_soft_reviews(REPOSITORY_ROOT) == []
