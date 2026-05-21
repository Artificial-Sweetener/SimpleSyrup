# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Domain models for graph-level image provenance tracing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

GraphLink: TypeAlias = tuple[str, int]


@dataclass(frozen=True)
class VaeDecodeProvenance:
    """Record the latent and VAE links that produced a decoded image."""

    decode_node_id: str
    image_output: GraphLink
    samples_link: GraphLink
    vae_link: GraphLink | None


@dataclass(frozen=True)
class BrokenProvenance:
    """Describe why an image link cannot be treated as an unmodified decode."""

    reason: str
    node_id: str | None = None
    class_type: str | None = None


@dataclass(frozen=True)
class PassthroughRule:
    """Describe an exact graph pass-through from output slot to input name."""

    input_name: str
