# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI nodes that switch between native normal and tiled VAE execution."""

from __future__ import annotations

from typing import Any

from ..runtime.vae_options_graph import ExpansionResult, VAEOptionsGraphBuilder
from . import tooltips

VAE_TILE_SIZE_DEFAULT = 512
VAE_TILE_SIZE_MIN = 64
VAE_TILE_SIZE_MAX = 4096
VAE_ENCODE_TILE_SIZE_STEP = 64
VAE_DECODE_TILE_SIZE_STEP = 32
VAE_OVERLAP_DEFAULT = 64
VAE_OVERLAP_MIN = 0
VAE_OVERLAP_MAX = 4096
VAE_OVERLAP_STEP = 32
VAE_TEMPORAL_SIZE_DEFAULT = 64
VAE_TEMPORAL_SIZE_MIN = 8
VAE_TEMPORAL_SIZE_MAX = 4096
VAE_TEMPORAL_SIZE_STEP = 4
VAE_TEMPORAL_OVERLAP_DEFAULT = 8
VAE_TEMPORAL_OVERLAP_MIN = 4
VAE_TEMPORAL_OVERLAP_MAX = 4096
VAE_TEMPORAL_OVERLAP_STEP = 4


class VAEEncodeOptions:
    """Encode images through ComfyUI's normal or tiled VAE encode nodes."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = (tooltips.VAE_OPTIONS_LATENT_OUTPUT,)
    FUNCTION = "encode"
    CATEGORY = "SimpleSyrup/Latent"
    DESCRIPTION = (
        "Encodes images to latent space with selectable normal or tiled VAE execution."
    )
    SEARCH_ALIASES = [
        "vae encode",
        "encode image",
        "tiled vae encode",
        "image to latent",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare VAE encode inputs and tiled execution controls."""

        return {
            "required": {
                "use_tiling": _use_tiling_input(),
                "pixels": (
                    "IMAGE",
                    {
                        "rawLink": True,
                        "tooltip": tooltips.VAE_OPTIONS_ENCODE_PIXELS,
                    },
                ),
                "vae": _vae_input(),
                "tile_size": _tile_size_input(VAE_ENCODE_TILE_SIZE_STEP),
                "overlap": _overlap_input(),
                "temporal_size": _temporal_size_input(
                    tooltips.VAE_OPTIONS_ENCODE_TEMPORAL_SIZE
                ),
                "temporal_overlap": _temporal_overlap_input(),
            },
        }

    def encode(
        self,
        use_tiling: bool,
        pixels: object,
        vae: object,
        tile_size: int,
        overlap: int,
        temporal_size: int = VAE_TEMPORAL_SIZE_DEFAULT,
        temporal_overlap: int = VAE_TEMPORAL_OVERLAP_DEFAULT,
    ) -> ExpansionResult:
        """Expand to ComfyUI's selected native VAE encode node."""

        return VAEOptionsGraphBuilder().build_encode(
            pixels=pixels,
            vae=vae,
            use_tiling=use_tiling,
            tile_size=tile_size,
            overlap=overlap,
            temporal_size=temporal_size,
            temporal_overlap=temporal_overlap,
        )


class VAEDecodeOptions:
    """Decode latents through ComfyUI's normal or tiled VAE decode nodes."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = (tooltips.VAE_OPTIONS_IMAGE_OUTPUT,)
    FUNCTION = "decode"
    CATEGORY = "SimpleSyrup/Latent"
    DESCRIPTION = (
        "Decodes latents to images with selectable normal or tiled VAE execution."
    )
    SEARCH_ALIASES = [
        "vae decode",
        "decode latent",
        "tiled vae decode",
        "latent to image",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare VAE decode inputs and tiled execution controls."""

        return {
            "required": {
                "use_tiling": _use_tiling_input(),
                "samples": (
                    "LATENT",
                    {
                        "rawLink": True,
                        "tooltip": tooltips.VAE_OPTIONS_DECODE_SAMPLES,
                    },
                ),
                "vae": _vae_input(),
                "tile_size": _tile_size_input(VAE_DECODE_TILE_SIZE_STEP),
                "overlap": _overlap_input(),
                "temporal_size": _temporal_size_input(
                    tooltips.VAE_OPTIONS_DECODE_TEMPORAL_SIZE
                ),
                "temporal_overlap": _temporal_overlap_input(),
            },
        }

    def decode(
        self,
        use_tiling: bool,
        samples: object,
        vae: object,
        tile_size: int,
        overlap: int,
        temporal_size: int = VAE_TEMPORAL_SIZE_DEFAULT,
        temporal_overlap: int = VAE_TEMPORAL_OVERLAP_DEFAULT,
    ) -> ExpansionResult:
        """Expand to ComfyUI's selected native VAE decode node."""

        return VAEOptionsGraphBuilder().build_decode(
            samples=samples,
            vae=vae,
            use_tiling=use_tiling,
            tile_size=tile_size,
            overlap=overlap,
            temporal_size=temporal_size,
            temporal_overlap=temporal_overlap,
        )


def _use_tiling_input() -> tuple[str, dict[str, object]]:
    """Return the shared tiling toggle declaration."""

    return (
        "BOOLEAN",
        {
            "default": False,
            "tooltip": tooltips.VAE_OPTIONS_USE_TILING,
        },
    )


def _vae_input() -> tuple[str, dict[str, object]]:
    """Return the shared raw-link VAE input declaration."""

    return (
        "VAE",
        {
            "rawLink": True,
            "tooltip": tooltips.VAE_OPTIONS_VAE,
        },
    )


def _tile_size_input(step: int) -> tuple[str, dict[str, object]]:
    """Return the tile-size input declaration for encode or decode."""

    return (
        "INT",
        {
            "default": VAE_TILE_SIZE_DEFAULT,
            "min": VAE_TILE_SIZE_MIN,
            "max": VAE_TILE_SIZE_MAX,
            "step": step,
            "advanced": True,
            "tooltip": tooltips.VAE_OPTIONS_TILE_SIZE,
        },
    )


def _overlap_input() -> tuple[str, dict[str, object]]:
    """Return the shared tile-overlap input declaration."""

    return (
        "INT",
        {
            "default": VAE_OVERLAP_DEFAULT,
            "min": VAE_OVERLAP_MIN,
            "max": VAE_OVERLAP_MAX,
            "step": VAE_OVERLAP_STEP,
            "advanced": True,
            "tooltip": tooltips.VAE_OPTIONS_OVERLAP,
        },
    )


def _temporal_size_input(tooltip: str) -> tuple[str, dict[str, object]]:
    """Return the shared temporal tile-size input declaration."""

    return (
        "INT",
        {
            "default": VAE_TEMPORAL_SIZE_DEFAULT,
            "min": VAE_TEMPORAL_SIZE_MIN,
            "max": VAE_TEMPORAL_SIZE_MAX,
            "step": VAE_TEMPORAL_SIZE_STEP,
            "advanced": True,
            "tooltip": tooltip,
        },
    )


def _temporal_overlap_input() -> tuple[str, dict[str, object]]:
    """Return the shared temporal overlap input declaration."""

    return (
        "INT",
        {
            "default": VAE_TEMPORAL_OVERLAP_DEFAULT,
            "min": VAE_TEMPORAL_OVERLAP_MIN,
            "max": VAE_TEMPORAL_OVERLAP_MAX,
            "step": VAE_TEMPORAL_OVERLAP_STEP,
            "advanced": True,
            "tooltip": tooltips.VAE_OPTIONS_TEMPORAL_OVERLAP,
        },
    )
