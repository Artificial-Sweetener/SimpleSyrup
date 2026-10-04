"""Expose original decoded-image latent recovery as a strict provenance bridge."""

from __future__ import annotations

from importlib import import_module
from typing import Any, ClassVar

from ..services.latent_provenance_service import LatentProvenanceService
from .sampler_options_schema import COMFY_IO, OptionsNodeBase


class LatentFromImageV3(OptionsNodeBase):
    """Return the original graph latent while keeping image pixels outside execution."""

    hidden: ClassVar[Any]

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the widget-free bridge with graph metadata supplied by Comfy."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.LatentFromImage",
            display_name="Latent From Image",
            category="SimpleSyrup/Latent",
            enable_expand=True,
            description=(
                "Recovers the original latent behind an unmodified "
                "decoded image using graph provenance."
            ),
            inputs=[
                COMFY_IO.Image.Input(
                    "image",
                    raw_link=True,
                    tooltip=(
                        "Connect an unmodified VAE Decode image; the original "
                        "latent is recovered through graph provenance."
                    ),
                )
            ],
            hidden=[COMFY_IO.Hidden.prompt],
            outputs=[
                COMFY_IO.Latent.Output(
                    "latent",
                    tooltip="Original sampler latent with its sampling provenance.",
                )
            ],
        )

    @classmethod
    def execute(cls, image: object) -> Any:
        """Expand to the original latent link without invoking a VAE encoder."""
        link = LatentProvenanceService().recover_link(
            cls.hidden.prompt, image, import_module("nodes").NODE_CLASS_MAPPINGS
        )
        return COMFY_IO.NodeOutput([link[0], link[1]], expand={})
