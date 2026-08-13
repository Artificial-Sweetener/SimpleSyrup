# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the isolated managed-Comfy launch profiles for U11 evidence."""

from __future__ import annotations

U11_CUSTOM_NODE_WHITELIST = (
    "SimpleSyrup",
    "SimpleSyrupBenchmarkProbe",
    "substitute-backend",
)


def sdxl_visual_sampling_launch_arguments() -> tuple[str, ...]:
    """Load only U11 runtime owners during real GPU sampling."""

    return (
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        *U11_CUSTOM_NODE_WHITELIST,
    )


def sdxl_visual_registration_launch_arguments() -> tuple[str, ...]:
    """Add CPU mode to the otherwise identical U11 registration profile."""

    return ("--cpu", *sdxl_visual_sampling_launch_arguments())
