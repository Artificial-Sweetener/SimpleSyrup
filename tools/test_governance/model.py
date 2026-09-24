# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable test-governance policy and reviewed state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from tools.architecture_governance.model import Diagnostic


@dataclass(frozen=True, slots=True)
class TestPolicy:
    """Define exact test discovery inputs and registry locations."""

    test_root: Path
    semantic_support_roots: tuple[Path, ...]
    root_source_extensions: frozenset[str]
    allowed_root_source_paths: frozenset[str]
    serial_policy: Path
    wait_calls: frozenset[str]
    wall_clock_calls: frozenset[str]
    xdist_environment_name: str
    repository_scratch_name: str
    debt_registry: Path
    waiver_registry: Path


@dataclass(frozen=True, slots=True)
class TestCandidate:
    """Identify one mechanically discovered fact requiring human review."""

    rule: str
    path: str
    locator: str
    evidence: str
    line: int

    @property
    def key(self) -> str:
        """Return the stable identity used by reviewed state records."""

        return f"{self.rule}|{self.path}|{self.locator}"


@dataclass(frozen=True, slots=True)
class TestDebt:
    """Describe reviewed test debt and its concrete remediation."""

    identifier: str
    owner: str
    rule: str
    candidates: tuple[str, ...]
    paths: tuple[str, ...]
    fingerprint: str
    issue: str
    review_by: date
    problem: str
    remediation: str


@dataclass(frozen=True, slots=True)
class TestWaiver:
    """Describe an exact classification or debt-remediation exception."""

    identifier: str
    owner: str
    kind: str
    disposition: str
    rule: str
    candidates: tuple[str, ...]
    paths: tuple[str, ...]
    fingerprint: str
    rationale: str
    issue: str
    review_by: date
    debt: str | None


@dataclass(frozen=True, slots=True)
class TestState:
    """Collect current test debt and waiver snapshots."""

    debts: tuple[TestDebt, ...]
    waivers: tuple[TestWaiver, ...]


@dataclass(frozen=True, slots=True)
class TestValidationResult:
    """Return discovered candidates together with policy diagnostics."""

    candidates: tuple[TestCandidate, ...]
    diagnostics: tuple[Diagnostic, ...]
