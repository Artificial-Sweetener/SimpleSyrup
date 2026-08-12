# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Align a derived Comfy CLIP with the model owned by its patcher."""

from __future__ import annotations

from typing import Protocol, cast


class _MutableClip(Protocol):
    """Describe the writable encoder reference on a derived host CLIP."""

    cond_stage_model: object


def align_clip_text_encoder_with_patcher(clip: object) -> None:
    """Make one derived CLIP execute the exact text encoder its patcher owns."""

    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        raise TypeError("Derived CLIP does not expose a patcher.")
    patcher_model = getattr(patcher, "model", None)
    if patcher_model is None:
        raise TypeError("Derived CLIP patcher does not expose a model.")
    text_encoder = getattr(clip, "cond_stage_model", None)
    if text_encoder is None:
        raise TypeError("Derived CLIP does not expose a text encoder.")
    if text_encoder is patcher_model:
        return

    mutable_clip = cast(_MutableClip, clip)
    try:
        mutable_clip.cond_stage_model = patcher_model
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "Derived CLIP text encoder cannot be aligned with its patcher model."
        ) from error
    if mutable_clip.cond_stage_model is not patcher_model:
        raise RuntimeError(
            "Derived CLIP did not retain its patcher-owned text encoder."
        )
