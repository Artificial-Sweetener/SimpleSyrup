# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for VAE Decode (Options)."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..nodes.vae_options import (
    VAE_DECODE_TILE_SIZE_STEP,
    VAE_OVERLAP_DEFAULT,
    VAE_OVERLAP_MAX,
    VAE_OVERLAP_MIN,
    VAE_OVERLAP_STEP,
    VAE_TEMPORAL_OVERLAP_DEFAULT,
    VAE_TEMPORAL_OVERLAP_MAX,
    VAE_TEMPORAL_OVERLAP_MIN,
    VAE_TEMPORAL_OVERLAP_STEP,
    VAE_TEMPORAL_SIZE_DEFAULT,
    VAE_TEMPORAL_SIZE_MAX,
    VAE_TEMPORAL_SIZE_MIN,
    VAE_TEMPORAL_SIZE_STEP,
    VAE_TILE_SIZE_DEFAULT,
    VAE_TILE_SIZE_MAX,
    VAE_TILE_SIZE_MIN,
    VAEDecodeOptions,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class VAEDecodeOptionsV3(_ComfyNodeBase):
    """Expose VAE Decode (Options) through Comfy's v3 extension API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the VAE Decode (Options) v3 schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.VAEDecodeOptions",
            display_name="VAE Decode (Options)",
            enable_expand=True,
            category="SimpleSyrup/Latent",
            description=VAEDecodeOptions.DESCRIPTION,
            search_aliases=VAEDecodeOptions.SEARCH_ALIASES,
            inputs=[
                _comfy_io.Boolean.Input(
                    "use_tiling",
                    default=False,
                    tooltip=tooltips.VAE_OPTIONS_USE_TILING,
                ),
                _comfy_io.Latent.Input(
                    "samples",
                    raw_link=True,
                    tooltip=tooltips.VAE_OPTIONS_DECODE_SAMPLES,
                ),
                _comfy_io.Vae.Input(
                    "vae",
                    raw_link=True,
                    tooltip=tooltips.VAE_OPTIONS_VAE,
                ),
                _comfy_io.Int.Input(
                    "tile_size",
                    default=VAE_TILE_SIZE_DEFAULT,
                    min=VAE_TILE_SIZE_MIN,
                    max=VAE_TILE_SIZE_MAX,
                    step=VAE_DECODE_TILE_SIZE_STEP,
                    advanced=True,
                    tooltip=tooltips.VAE_OPTIONS_TILE_SIZE,
                ),
                _comfy_io.Int.Input(
                    "overlap",
                    default=VAE_OVERLAP_DEFAULT,
                    min=VAE_OVERLAP_MIN,
                    max=VAE_OVERLAP_MAX,
                    step=VAE_OVERLAP_STEP,
                    advanced=True,
                    tooltip=tooltips.VAE_OPTIONS_OVERLAP,
                ),
                _comfy_io.Int.Input(
                    "temporal_size",
                    default=VAE_TEMPORAL_SIZE_DEFAULT,
                    min=VAE_TEMPORAL_SIZE_MIN,
                    max=VAE_TEMPORAL_SIZE_MAX,
                    step=VAE_TEMPORAL_SIZE_STEP,
                    advanced=True,
                    tooltip=tooltips.VAE_OPTIONS_DECODE_TEMPORAL_SIZE,
                ),
                _comfy_io.Int.Input(
                    "temporal_overlap",
                    default=VAE_TEMPORAL_OVERLAP_DEFAULT,
                    min=VAE_TEMPORAL_OVERLAP_MIN,
                    max=VAE_TEMPORAL_OVERLAP_MAX,
                    step=VAE_TEMPORAL_OVERLAP_STEP,
                    advanced=True,
                    tooltip=tooltips.VAE_OPTIONS_TEMPORAL_OVERLAP,
                ),
            ],
            outputs=[
                _comfy_io.Image.Output(
                    "image",
                    tooltip=tooltips.VAE_OPTIONS_IMAGE_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        use_tiling: bool,
        samples: object,
        vae: object,
        tile_size: int,
        overlap: int,
        temporal_size: int,
        temporal_overlap: int,
    ) -> Any:
        """Expand through the legacy VAE Decode (Options) implementation."""

        return VAEDecodeOptions().decode(
            use_tiling=use_tiling,
            samples=samples,
            vae=vae,
            tile_size=tile_size,
            overlap=overlap,
            temporal_size=temporal_size,
            temporal_overlap=temporal_overlap,
        )
