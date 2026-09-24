# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Enforce SimpleSyrup package dependency direction with exact debt snapshots."""

from __future__ import annotations

import ast
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypedDict, cast

from .metrics import source_fingerprint
from .model import Diagnostic

IMPORT_DEBT_PATH = Path("governance/architecture/import_debt.toml")
_FORBIDDEN_EDGES = frozenset(
    {
        ("domain", "masking"),
        ("domain", "integration"),
        ("domain", "runtime"),
        ("domain", "services"),
        ("image", "integration"),
        ("masking", "integration"),
        ("nodes", "integration"),
        ("nodes_v3", "integration"),
        ("runtime", "integration"),
        ("runtime", "nodes"),
        ("runtime", "nodes_v3"),
        ("runtime", "services"),
        ("services", "integration"),
        ("services", "nodes"),
        ("services", "nodes_v3"),
        ("shared", "domain"),
        ("shared", "integration"),
        ("shared", "runtime"),
        ("shared", "services"),
    }
)


class _ImportDebt(TypedDict):
    """Describe one exact source file's known forbidden imports."""

    id: str
    path: str
    imports: list[str]
    fingerprint: str
    issue: str
    review_by: date
    problem: str
    remediation: str


def validate_import_boundaries(
    root: Path, *, today: date | None = None
) -> list[Diagnostic]:
    """Return diagnostics for unreviewed or stale package dependency debt."""

    registry_path = root / IMPORT_DEBT_PATH
    try:
        payload = tomllib.loads(registry_path.read_text(encoding="utf-8"))
        records = cast(list[_ImportDebt], payload.get("debts", []))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return [Diagnostic("IMPORTSTATE001", IMPORT_DEBT_PATH.as_posix(), str(error))]
    current = _forbidden_imports(root)
    diagnostics: list[Diagnostic] = []
    recorded_paths: set[str] = set()
    current_date = today or datetime.now(UTC).date()
    for record in records:
        path = record["path"]
        if path in recorded_paths:
            diagnostics.append(
                Diagnostic(
                    "IMPORTSTATE002",
                    IMPORT_DEBT_PATH.as_posix(),
                    f"forbidden-import path {path} has multiple debt records",
                )
            )
            continue
        recorded_paths.add(path)
        expected = tuple(record["imports"])
        if expected != tuple(sorted(expected)) or current.get(path) != expected:
            diagnostics.append(
                Diagnostic(
                    "IMPORTDEBT001",
                    IMPORT_DEBT_PATH.as_posix(),
                    f"debt {record['id']} must match the exact sorted forbidden "
                    f"imports for {path}",
                )
            )
        elif source_fingerprint(root, (path,)) != record["fingerprint"]:
            diagnostics.append(
                Diagnostic(
                    "IMPORTDEBT002",
                    IMPORT_DEBT_PATH.as_posix(),
                    f"debt {record['id']} source changed; repeat ownership review",
                )
            )
        if record["review_by"] < current_date:
            diagnostics.append(
                Diagnostic(
                    "IMPORTDEBT003",
                    IMPORT_DEBT_PATH.as_posix(),
                    f"debt {record['id']} expired on {record['review_by'].isoformat()}",
                )
            )
    for path, imports in current.items():
        if path not in recorded_paths:
            diagnostics.append(
                Diagnostic(
                    "IMPORT001",
                    path,
                    "forbidden package dependency requires an exact debt record: "
                    + ", ".join(imports),
                )
            )
    for path in recorded_paths - current.keys():
        diagnostics.append(
            Diagnostic(
                "IMPORTDEBT004",
                IMPORT_DEBT_PATH.as_posix(),
                f"recorded forbidden-import debt for {path} is resolved or stale",
            )
        )
    return sorted(diagnostics, key=lambda item: (item.path, item.rule, item.message))


def _forbidden_imports(root: Path) -> dict[str, tuple[str, ...]]:
    """Discover exact first-party imports that point against layer direction."""

    package_root = root / "simple_syrup"
    findings: dict[str, tuple[str, ...]] = {}
    for path in sorted(package_root.rglob("*.py")):
        if "third_party" in path.parts:
            continue
        relative = path.relative_to(package_root)
        source_layer = relative.parts[0]
        modules = {
            module
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            for module in _imported_modules(node, relative)
            if module.startswith("simple_syrup.")
            and (source_layer, module.split(".", maxsplit=2)[1]) in _FORBIDDEN_EDGES
        }
        if modules:
            findings[path.relative_to(root).as_posix()] = tuple(sorted(modules))
    return findings


def _imported_modules(node: ast.AST, relative_path: Path) -> tuple[str, ...]:
    """Resolve absolute and package-relative imports for one syntax node."""

    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    if node.level == 0:
        return (node.module,) if node.module else ()
    package_parts = list(relative_path.with_suffix("").parts[:-1])
    retained = package_parts[: len(package_parts) - (node.level - 1)]
    suffix = node.module.split(".") if node.module else []
    return (".".join(("simple_syrup", *retained, *suffix)),)
