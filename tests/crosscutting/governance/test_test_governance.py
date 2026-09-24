# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify capability placement and reviewed test-governance state."""

from __future__ import annotations

from pathlib import Path

from support.repository import REPOSITORY_ROOT

from tools.test_governance.discovery import (
    LAYOUT_RULE,
    TYPESCRIPT_LAYOUT_RULE,
    TYPESCRIPT_OPTIONAL_RULE,
    discover_test_candidates,
)
from tools.test_governance.loading import load_test_policy
from tools.test_governance.validation import validate_test_governance


def _write(path: Path, content: str) -> None:
    """Write one isolated test-governance fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_fixture(root: Path) -> None:
    """Create the smallest complete test-governance repository."""

    _write(
        root / "governance/testing/policy.toml",
        """schema_version = 1
[scope]
test_root = "tests"
semantic_support_roots = ["tools"]
root_source_extensions = [".py", ".pyi"]
allowed_root_source_paths = ["tests/conftest.py", "tests/ci_test_policy.py"]
[discovery]
serial_policy = "tests/ci_test_policy.py"
wait_calls = ["QTest.qWait", "time.sleep"]
wall_clock_calls = ["time.perf_counter"]
xdist_environment_name = "PYTEST_XDIST_WORKER"
repository_scratch_name = ".pytest-tmp"
[registries]
debt = "governance/testing/debt.toml"
waivers = "governance/testing/waivers.toml"
""",
    )
    _write(root / "governance/testing/debt.toml", "schema_version = 1\ndebts = []\n")
    _write(
        root / "governance/testing/waivers.toml",
        "schema_version = 1\nwaivers = []\n",
    )
    _write(
        root / "tests/ci_test_policy.py",
        "ISOLATED_TEST_MODULES = ()\nSERIAL_TEST_MODULES = ()\n",
    )
    _write(root / "tests/conftest.py", "")
    _write(root / "tools/__init__.py", "")


def test_discovery_covers_python_and_frontend_root_layout(tmp_path: Path) -> None:
    """Reject authored tests left outside capability owners."""

    _write_fixture(tmp_path)
    _write(tmp_path / "tests/test_root.py", "def test_root() -> None: pass\n")
    _write(tmp_path / "web/tests/root.test.ts", "test('root', () => {});\n")
    policy = load_test_policy(tmp_path / "governance/testing/policy.toml")

    rules = {item.rule for item in discover_test_candidates(tmp_path, policy)}

    assert {LAYOUT_RULE, TYPESCRIPT_LAYOUT_RULE} <= rules


def test_discovery_reports_optional_frontend_proof(tmp_path: Path) -> None:
    """Require review of skipped, todo, or exclusive frontend proof."""

    _write_fixture(tmp_path)
    _write(
        tmp_path / "web/tests/media/preview.test.ts",
        "describe.only('preview', () => {});\n",
    )
    policy = load_test_policy(tmp_path / "governance/testing/policy.toml")

    rules = {item.rule for item in discover_test_candidates(tmp_path, policy)}

    assert TYPESCRIPT_OPTIONAL_RULE in rules


def test_current_repository_has_exact_reviewed_test_state() -> None:
    """Keep every reliability candidate exactly reviewed."""

    result = validate_test_governance(REPOSITORY_ROOT)

    assert not [item for item in result.diagnostics if item.severity == "error"]
