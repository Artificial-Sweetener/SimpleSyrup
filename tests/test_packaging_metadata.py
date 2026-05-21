# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for package metadata and node-local requirements."""

from __future__ import annotations

import json
import tomllib
from importlib import import_module
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
EXPECTED_RUNTIME_REQUIREMENTS = (
    "torchlanc",
    "ultralytics",
    "onnxruntime",
    "segment-anything",
    "timm",
    "addict",
    "yapf",
    "huggingface-hub",
)


def _requirements(path: Path) -> tuple[Requirement, ...]:
    """Parse non-comment requirement lines from a requirements file."""

    return tuple(
        Requirement(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_requirements_are_node_local_and_do_not_duplicate_comfyui() -> None:
    """Keep SimpleSyrup requirements limited to packages ComfyUI does not own."""

    simple_syrup_requirements = _requirements(REPO_ROOT / "requirements.txt")
    comfy_requirements = _requirements(COMFY_ROOT / "requirements.txt")
    comfy_names = {requirement.name.lower() for requirement in comfy_requirements}

    assert (
        tuple(requirement.name.lower() for requirement in simple_syrup_requirements)
        == EXPECTED_RUNTIME_REQUIREMENTS
    )
    assert not {
        requirement.name.lower()
        for requirement in simple_syrup_requirements
        if requirement.name.lower() in comfy_names
    }


def test_pyproject_reads_runtime_dependencies_from_requirements_txt() -> None:
    """Use requirements.txt as the only runtime dependency source."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))

    assert pyproject["project"]["dynamic"] == ["dependencies"]
    assert "dependencies" not in pyproject["project"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["dependencies"] == {
        "file": ["requirements.txt"]
    }


def test_pyproject_has_registry_ready_metadata() -> None:
    """Require Comfy Registry and package metadata needed for publication."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))

    assert pyproject["build-system"]["requires"] == ["setuptools>=77"]
    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["project"]["license"] == "AGPL-3.0-or-later"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    assert (
        pyproject["project"]["description"]
        == "Workflow-focused ComfyUI extensions for image generation."
    )
    assert pyproject["project"]["urls"]["Repository"].endswith("/SimpleSyrup")
    assert pyproject["tool"]["comfy"]["PublisherId"] == "artificialsweetener"
    assert pyproject["tool"]["comfy"]["DisplayName"] == "SimpleSyrup"
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "simple_syrup*"
    ]


def test_frontend_package_metadata_matches_python_package_license() -> None:
    """Keep frontend package metadata aligned with the Python package license."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    package = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))

    assert package["private"] is True
    assert package["license"] == pyproject["project"]["license"]


def test_package_versions_match_release_metadata() -> None:
    """Keep release-managed version fields aligned across package metadata."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    package = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads(
        (REPO_ROOT / "package-lock.json").read_text(encoding="utf-8")
    )
    simple_syrup = import_module("simple_syrup")

    expected_version = pyproject["project"]["version"]

    assert package["version"] == expected_version
    assert package_lock["version"] == expected_version
    assert package_lock["packages"][""]["version"] == expected_version
    assert simple_syrup.__version__ == expected_version


def test_frontend_dist_bundle_is_tracked_for_comfy_serving() -> None:
    """ComfyUI serves the checked-in frontend bundle from WEB_DIRECTORY."""

    dist_bundle = REPO_ROOT / "web" / "dist" / "simple-syrup.js"

    assert dist_bundle.is_file()
