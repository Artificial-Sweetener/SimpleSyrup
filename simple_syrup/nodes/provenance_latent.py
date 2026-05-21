# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI nodes that reuse latent provenance behind decoded images."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any, TypedDict, cast

from ..domain.graph_provenance import BrokenProvenance, GraphLink, VaeDecodeProvenance
from ..runtime.comfy_graph_provenance import (
    ProvenanceTrace,
    links_match,
    trace_vae_decode_provenance,
)

LATENT_PROVENANCE_ERROR = (
    "Unable to find an unmodified VAE Decode source for this image. Connect an "
    "image that is directly decoded from a latent, or place this node immediately "
    "after a transparent cube boundary whose image output comes from VAE Decode."
)


class ExpansionResult(TypedDict):
    """ComfyUI dynamic expansion result returned by provenance-aware nodes."""

    expand: dict[str, dict[str, Any]]
    result: tuple[list[object], ...]


class SimpleVAEEncode:
    """Encode images while reusing source latents when provenance proves safety."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = (
        "Latent recovered from the decoded image source or produced by normal "
        "VAE encoding.",
    )
    FUNCTION = "encode"
    CATEGORY = "SimpleSyrup/Latent"
    DESCRIPTION = (
        "Encodes an image to latent space, reusing the source latent when the image "
        "is proven to be an unmodified VAE decode."
    )
    SEARCH_ALIASES = [
        "vae encode",
        "encode image",
        "recover latent",
        "reuse decoded latent",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Declare the provenance-aware VAE encode input contract."""

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "rawLink": True,
                        "tooltip": (
                            "Image to encode. If it comes directly from VAE Decode "
                            "through transparent pass-through nodes, the original "
                            "latent is reused; edited or loaded images are encoded "
                            "normally."
                        ),
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "rawLink": True,
                        "tooltip": (
                            "VAE used for normal image encoding. Latent reuse is "
                            "used only when this VAE matches the VAE that decoded "
                            "the source image."
                        ),
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    def encode(
        self,
        image: object,
        vae: object,
        prompt: Mapping[str, Any] | None = None,
    ) -> ExpansionResult:
        """Return the source latent when safe, otherwise expand to `VAEEncode`."""

        provenance = _trace_prompt_provenance(prompt, image)
        if (
            isinstance(provenance, VaeDecodeProvenance)
            and provenance.vae_link is not None
            and links_match(vae, provenance.vae_link)
        ):
            return _link_result(provenance.samples_link)

        return _vae_encode_expansion(image, vae)


class UpscaleLatentFromImage:
    """Upscale the latent behind an unmodified decoded image."""

    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = (
        "Upscaled latent produced from the source latent behind the decoded image.",
    )
    FUNCTION = "upscale"
    CATEGORY = "SimpleSyrup/Latent"
    DESCRIPTION = "Upscales the latent that produced an unmodified decoded image."
    SEARCH_ALIASES = [
        "upscale latent",
        "latent upscale from image",
        "decoded image latent upscale",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Declare the latent-upscale-from-image input contract."""

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "rawLink": True,
                        "tooltip": (
                            "Connect an image that comes directly from VAE Decode, "
                            "with only transparent pass-through nodes between them. "
                            "Pixel edits, image upscalers, crops, detailers, "
                            "loaders, and previews break latent provenance."
                        ),
                    },
                ),
                "upscale_method": (
                    cls.UPSCALE_METHODS,
                    {
                        "tooltip": (
                            "Interpolation method passed to Comfy's Upscale Latent "
                            "By behavior."
                        )
                    },
                ),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.01,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": (
                            "Multiplier for the latent width and height passed to "
                            "Comfy's Upscale Latent By behavior."
                        ),
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    def upscale(
        self,
        image: object,
        upscale_method: str,
        scale_factor: float,
        prompt: Mapping[str, Any] | None = None,
    ) -> ExpansionResult:
        """Expand to Comfy's `LatentUpscaleBy` when provenance is valid."""

        provenance = _trace_prompt_provenance(prompt, image)
        if isinstance(provenance, BrokenProvenance):
            raise ValueError(LATENT_PROVENANCE_ERROR)
        return _latent_upscale_by_expansion(
            provenance.samples_link,
            upscale_method,
            scale_factor,
        )


def _trace_prompt_provenance(
    prompt: Mapping[str, Any] | None,
    image: object,
) -> ProvenanceTrace:
    """Trace an image link with the active ComfyUI node registry."""

    if prompt is None:
        return BrokenProvenance("prompt metadata is unavailable")
    return trace_vae_decode_provenance(prompt, image, _node_registry())


def _link_result(link: GraphLink) -> ExpansionResult:
    """Return an existing graph link through a dynamic expansion result."""

    return {"expand": {}, "result": (_to_comfy_link(link),)}


def _vae_encode_expansion(image: object, vae: object) -> ExpansionResult:
    """Build a dynamic `VAEEncode` fallback graph."""

    builder = _graph_builder()
    encoded = builder.node(
        "VAEEncode",
        pixels=_graph_value(image),
        vae=_graph_value(vae),
    )
    return {"expand": builder.finalize(), "result": (encoded.out(0),)}


def _latent_upscale_by_expansion(
    samples_link: GraphLink,
    upscale_method: str,
    scale_factor: float,
) -> ExpansionResult:
    """Build a dynamic `LatentUpscaleBy` graph around the source latent."""

    builder = _graph_builder()
    upscaled = builder.node(
        "LatentUpscaleBy",
        samples=_to_comfy_link(samples_link),
        upscale_method=upscale_method,
        scale_by=scale_factor,
    )
    return {"expand": builder.finalize(), "result": (upscaled.out(0),)}


def _graph_value(value: object) -> object:
    """Normalize graph-link values while preserving regular fallback values."""

    if isinstance(value, tuple):
        link = _tuple_link(value)
        if link is not None:
            return _to_comfy_link(link)
    return value


def _tuple_link(value: tuple[object, ...]) -> GraphLink | None:
    """Return a typed graph link from a tuple value when possible."""

    if len(value) != 2:
        return None
    node_id, output_slot = value
    if not isinstance(node_id, str) or not isinstance(output_slot, int):
        return None
    return (node_id, output_slot)


def _to_comfy_link(link: GraphLink) -> list[object]:
    """Convert an internal graph link tuple to Comfy's serialized list shape."""

    return [link[0], link[1]]


def _node_registry() -> Mapping[str, type[object]]:
    """Return ComfyUI's active node registry without importing it at module load."""

    nodes_module = import_module("nodes")
    return cast(
        Mapping[str, type[object]],
        nodes_module.NODE_CLASS_MAPPINGS,
    )


def _graph_builder() -> Any:
    """Return Comfy's dynamic graph builder without import-time Comfy coupling."""

    graph_utils = import_module("comfy_execution.graph_utils")
    graph_builder = cast(Any, graph_utils.GraphBuilder)
    return graph_builder()
