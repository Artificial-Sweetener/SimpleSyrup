# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select explicitly ordered subsets from SDXL visual case families."""

from __future__ import annotations

from .visual_case_model import SdxlVisualCase


def select_visual_cases(
    available: tuple[SdxlVisualCase, ...],
    case_ids: tuple[str, ...],
) -> tuple[SdxlVisualCase, ...]:
    """Return an explicitly ordered subset or the complete available family."""

    if not case_ids:
        return available
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Visual case ids must be unique.")
    by_id = {case.case_id: case for case in available}
    unknown = tuple(case_id for case_id in case_ids if case_id not in by_id)
    if unknown:
        raise ValueError(f"Unknown visual case ids: {unknown!r}.")
    return tuple(by_id[case_id] for case_id in case_ids)
