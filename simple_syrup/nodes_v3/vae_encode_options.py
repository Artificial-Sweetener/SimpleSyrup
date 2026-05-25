# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for VAE Encode (Options)."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..nodes.vae_options import (
    VAE_ENCODE_TILE_SIZE_STEP,
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
    VAEEncodeOptions,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class VAEEncodeOptionsV3(_ComfyNodeBase):
    """Expose VAE Encode (Options) through Comfy's v3 extension API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the VAE Encode (Options) v3 schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.VAEEncodeOptions",
            display_name="VAE Encode (Options)",
            enable_expand=True,
            category="SimpleSyrup/Latent",
            description=VAEEncodeOptions.DESCRIPTION,
            search_aliases=VAEEncodeOptions.SEARCH_ALIASES,
            inputs=[
                _comfy_io.Boolean.Input(
                    "use_tiling",
                    default=False,
                    tooltip=tooltips.VAE_OPTIONS_USE_TILING,
                ),
                _comfy_io.Image.Input(
                    "pixels",
                    raw_link=True,
                    tooltip=tooltips.VAE_OPTIONS_ENCODE_PIXELS,
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
                    step=VAE_ENCODE_TILE_SIZE_STEP,
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
                    tooltip=tooltips.VAE_OPTIONS_ENCODE_TEMPORAL_SIZE,
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
                _comfy_io.Latent.Output(
                    "latent",
                    tooltip=tooltips.VAE_OPTIONS_LATENT_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        use_tiling: bool,
        pixels: object,
        vae: object,
        tile_size: int,
        overlap: int,
        temporal_size: int,
        temporal_overlap: int,
    ) -> Any:
        """Expand through the legacy VAE Encode (Options) implementation."""

        return VAEEncodeOptions().encode(
            use_tiling=use_tiling,
            pixels=pixels,
            vae=vae,
            tile_size=tile_size,
            overlap=overlap,
            temporal_size=temporal_size,
            temporal_overlap=temporal_overlap,
        )
