# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the immutable spatial execution vocabulary for regional attention."""

from enum import StrEnum


class RegionalAttentionExecutionMode(StrEnum):
    """Identify how one prepared regional model traverses the latent canvas."""

    FULL = "full-context"
    TILED = "tiled"
    CONTEXTUAL = "Contextual"
