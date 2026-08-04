# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve decoded image geometry from ComfyUI latent runtime metadata."""

from __future__ import annotations

from typing import Any


def decoded_image_dimensions(
    *,
    model: Any,
    latent_image: dict[str, Any],
    latent_height: int,
    latent_width: int,
) -> tuple[int, int]:
    """Return decoded height and width using ComfyUI's spatial latent ratio."""

    ratio = latent_image.get("downscale_ratio_spacial")
    if ratio is None:
        get_model_object = getattr(model, "get_model_object", None)
        if not callable(get_model_object):
            raise ValueError(
                "Context SEGS requires SEGS input dimensions or a model exposing "
                "its latent format."
            )
        latent_format = get_model_object("latent_format")
        ratio = getattr(latent_format, "spacial_downscale_ratio", None)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 0:
        raise ValueError(
            "Context SEGS requires a positive latent spatial downscale ratio."
        )
    return round(latent_height * ratio), round(latent_width * ratio)
