# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify isolated managed-Comfy launch profiles for U11."""

from __future__ import annotations

from tools.sdxl_attention_coupling_integration.visual_launch import (
    U11_CUSTOM_NODE_WHITELIST,
    sdxl_visual_registration_launch_arguments,
    sdxl_visual_sampling_launch_arguments,
)


def test_sampling_profile_whitelists_only_exact_u11_owners() -> None:
    """Exclude unrelated custom-node import behavior from visual evidence."""

    arguments = sdxl_visual_sampling_launch_arguments()

    assert arguments[:2] == (
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
    )
    assert arguments[2:] == U11_CUSTOM_NODE_WHITELIST
    assert "--cpu" not in arguments


def test_registration_profile_adds_only_cpu_mode() -> None:
    """Keep registration and sampling custom-node surfaces identical."""

    assert sdxl_visual_registration_launch_arguments() == (
        "--cpu",
        *sdxl_visual_sampling_launch_arguments(),
    )
