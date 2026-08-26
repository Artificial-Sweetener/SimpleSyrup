# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve explicit two-dimensional geometry for flattened attention maps."""

from __future__ import annotations

import math


def factor_spatial_geometry(
    token_count: int, *, target_aspect: float
) -> tuple[int, int]:
    """Return the token factor pair nearest the supplied positive aspect ratio."""

    if type(token_count) is not int or token_count < 1:
        raise ValueError("Attention spatial token count must be positive.")
    if target_aspect <= 0.0:
        raise ValueError("Attention target aspect ratio must be positive.")
    candidates: list[tuple[float, int, int]] = []
    for height in range(1, math.isqrt(token_count) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for candidate_height, candidate_width in ((height, width), (width, height)):
            error = abs(math.log((candidate_width / candidate_height) / target_aspect))
            candidates.append((error, candidate_height, candidate_width))
    _error, height, width = min(candidates)
    return height, width
