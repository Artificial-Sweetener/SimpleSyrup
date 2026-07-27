# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Carry hooked global conditioning alongside one regional conditioning entry."""

from __future__ import annotations

from typing import Any, TypeAlias

Conditioning: TypeAlias = list[list[Any]]
_GLOBAL_COMPANION_KEY = "simple_syrup.regional_global_companion"


def attach_global_companion(
    conditioning: object,
    global_conditioning: object,
) -> Conditioning:
    """Attach one hooked global conditioning without changing Comfy's public type."""

    local = _validate_conditioning(conditioning, "regional conditioning")
    companion = _validate_conditioning(
        global_conditioning,
        "regional global companion",
    )
    attached = _copy_conditioning(local)
    attached[0][1][_GLOBAL_COMPANION_KEY] = _copy_conditioning(companion)
    return attached


def detach_global_companion(
    conditioning: Conditioning,
) -> tuple[Conditioning, Conditioning | None]:
    """Remove and return an attached global companion before Comfy sampling."""

    detached = _copy_conditioning(conditioning)
    companion = detached[0][1].pop(_GLOBAL_COMPANION_KEY, None)
    if companion is None:
        return detached, None
    return detached, _validate_conditioning(
        companion,
        "regional global companion",
    )


def _validate_conditioning(value: object, name: str) -> Conditioning:
    """Validate the Comfy conditioning container used by the internal graph node."""

    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty CONDITIONING value.")
    for index, item in enumerate(value):
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not isinstance(item[1], dict)
        ):
            raise ValueError(f"{name} item {index} must contain a tensor and metadata.")
    return value


def _copy_conditioning(conditioning: Conditioning) -> Conditioning:
    """Copy conditioning containers and metadata without cloning tensors."""

    return [[item[0], dict(item[1])] for item in conditioning]
