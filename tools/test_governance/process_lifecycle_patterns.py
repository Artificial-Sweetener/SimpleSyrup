# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discover child-process lifetime ownership risks in tests."""

from __future__ import annotations

import ast

from .ast_analysis import call_name
from .model import TestCandidate

CHILD_PROCESS_RULE = "PROCESS001"


def process_lifecycle_pattern_candidates(
    *,
    path: str,
    tree: ast.Module,
    aliases: dict[str, str],
) -> list[TestCandidate]:
    """Find child processes whose lifetime is not owned by a context manager."""

    parent_by_node = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    candidates: list[TestCandidate] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and call_name(node.func, aliases) == "subprocess.Popen"
        ):
            continue
        parent = parent_by_node.get(node)
        context_managed = (
            isinstance(parent, ast.withitem) and parent.context_expr is node
        )
        if context_managed:
            continue
        candidates.append(
            TestCandidate(
                rule=CHILD_PROCESS_RULE,
                path=path,
                locator=f"<module>:unscoped-child-process:{len(candidates) + 1}",
                evidence=(
                    "starts a child process without a context-managed lifetime; "
                    "termination and bounded cleanup require source review"
                ),
                line=node.lineno,
            )
        )
    return candidates


__all__ = ["CHILD_PROCESS_RULE", "process_lifecycle_pattern_candidates"]
