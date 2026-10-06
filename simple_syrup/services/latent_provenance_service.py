# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recover original latents from image graph provenance without pixel encoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..domain.graph_provenance import BrokenProvenance, GraphLink
from ..runtime.comfy_graph_provenance import NodeRegistry, trace_vae_decode_provenance


class LatentProvenanceService:
    """Admit only unmodified image paths backed by a VAE decode source."""

    def recover_link(
        self, prompt: Mapping[str, Any], image: object, registry: NodeRegistry
    ) -> GraphLink:
        """Return the original samples link or a specific actionable graph error."""
        provenance = trace_vae_decode_provenance(prompt, image, registry)
        if isinstance(provenance, BrokenProvenance):
            raise ValueError(
                f"Latent From Image requires an unmodified VAE Decode "
                f"source: {provenance.reason}. Connect the original "
                f"decoded image or a direct latent connection."
            )
        return provenance.samples_link
