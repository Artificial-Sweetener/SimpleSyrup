# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate exact human review of every current soft-ceiling source file."""

from __future__ import annotations

import tomllib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypedDict, cast

from .loading import load_policy
from .metrics import governed_source_paths, production_line_count, source_fingerprint
from .model import Diagnostic

SOFT_REVIEW_PATH = Path("governance/architecture/soft_reviews.toml")


class _Remediation(TypedDict):
    """Describe one reviewed soft-ceiling mixed-ownership finding."""

    id: str
    path: str
    responsibilities: list[str]
    next_extraction: str


def validate_soft_reviews(root: Path, *, today: date | None = None) -> list[Diagnostic]:
    """Require an exact current disposition for every soft-ceiling file."""

    registry = root / SOFT_REVIEW_PATH
    try:
        payload = tomllib.loads(registry.read_text(encoding="utf-8"))
        policy = load_policy(root / "governance/architecture/policy.toml")
        cohesive = tuple(cast(list[str], payload["cohesive_paths"]))
        debt = tuple(cast(list[str], payload["debt_paths"]))
        remediations = cast(list[_Remediation], payload["remediations"])
        fingerprint = cast(str, payload["fingerprint"])
        review_by = cast(date, payload["review_by"])
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        return [Diagnostic("SOFTSTATE001", SOFT_REVIEW_PATH.as_posix(), str(error))]
    current = tuple(
        path.relative_to(root).as_posix()
        for path in governed_source_paths(root, policy)
        if policy.soft_lines < production_line_count(path) <= policy.hard_lines
    )
    diagnostics: list[Diagnostic] = []
    reviewed = tuple(sorted((*cohesive, *debt)))
    if cohesive != tuple(sorted(cohesive)) or debt != tuple(sorted(debt)):
        diagnostics.append(
            Diagnostic(
                "SOFTSTATE002",
                SOFT_REVIEW_PATH.as_posix(),
                "cohesive_paths and debt_paths must use stable sorted order",
            )
        )
    if set(cohesive) & set(debt) or reviewed != current:
        diagnostics.append(
            Diagnostic(
                "SOFTSTATE003",
                SOFT_REVIEW_PATH.as_posix(),
                "soft reviews must classify every exact current warning once",
            )
        )
    elif source_fingerprint(root, reviewed) != fingerprint:
        diagnostics.append(
            Diagnostic(
                "SOFTSTATE004",
                SOFT_REVIEW_PATH.as_posix(),
                "soft-ceiling source changed; repeat human ownership review",
            )
        )
    if review_by < (today or datetime.now(UTC).date()):
        diagnostics.append(
            Diagnostic(
                "SOFTSTATE005",
                SOFT_REVIEW_PATH.as_posix(),
                f"soft-ceiling review expired on {review_by.isoformat()}",
            )
        )
    remediation_paths = tuple(sorted(item["path"] for item in remediations))
    if remediation_paths != debt or len(remediation_paths) != len(
        set(remediation_paths)
    ):
        diagnostics.append(
            Diagnostic(
                "SOFTDEBT001",
                SOFT_REVIEW_PATH.as_posix(),
                "every soft debt path requires exactly one remediation record",
            )
        )
    for item in remediations:
        if len(item["responsibilities"]) < 2 or len(item["next_extraction"]) < 80:
            diagnostics.append(
                Diagnostic(
                    "SOFTDEBT002",
                    SOFT_REVIEW_PATH.as_posix(),
                    f"remediation {item['id']} requires substantive source review",
                )
            )
    return diagnostics
