# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node registration for SimpleSyrup."""

from __future__ import annotations

from ..runtime.prompt_control_availability import prompt_control_is_available


def get_nodes() -> list[type[object]]:
    """Return v3 nodes that can be advertised in this environment."""

    from .scale_factor import ScaleFactorV3
    from .simple_load_checkpoint import SimpleLoadCheckpointV3
    from .tile_and_tag_segs import TileAndTagSEGSV3
    from .vae_decode_options import VAEDecodeOptionsV3
    from .vae_encode_options import VAEEncodeOptionsV3
    from .wd14_tagger_loader import WD14TaggerLoaderV3

    if not prompt_control_is_available():
        return [
            WD14TaggerLoaderV3,
            TileAndTagSEGSV3,
            SimpleLoadCheckpointV3,
            ScaleFactorV3,
            VAEDecodeOptionsV3,
            VAEEncodeOptionsV3,
        ]

    from .encode_prompt_batch_with_prompt_control import (
        EncodePromptBatchWithPromptControl,
    )
    from .schedule_and_encode_prompts_with_prompt_control import (
        ScheduleAndEncodePromptsWithPromptControl,
    )

    return [
        WD14TaggerLoaderV3,
        TileAndTagSEGSV3,
        SimpleLoadCheckpointV3,
        ScaleFactorV3,
        VAEDecodeOptionsV3,
        VAEEncodeOptionsV3,
        EncodePromptBatchWithPromptControl,
        ScheduleAndEncodePromptsWithPromptControl,
    ]


__all__ = ["get_nodes"]
